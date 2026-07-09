"""
Semantic metric ablations (entropy / density) and quality thresholds for CEO follow-up.

Entropy / density toggles (default: both on):
  export HASHIRU_ENABLE_SEMANTIC_ENTROPY=0
  export HASHIRU_ENABLE_SEMANTIC_DENSITY=0

When both are off, local Ollama workers skip the HTTP logprobs chat path and use the library
``ollama.chat`` API instead (avoids HASHIRU_OLLAMA_CHAT_TIMEOUT stalls on long R1-style runs).

Multi-agent tool (default: on). When off, AskMultipleAgents is not exposed to the CEO and calls are rejected:
  export HASHIRU_ENABLE_ASK_MULTIPLE_AGENTS=0

Thresholds (used only for metrics that remain enabled). When violated, AskAgent returns
semantic_quality_concern=True so the CEO decides the next step (reprompt, other agent,
AskMultipleAgents, retire worker, etc.). The worker is NOT auto re-prompted.

  export HASHIRU_SEMANTIC_ENTROPY_THRESHOLD=1.65   # reprompt concern if entropy ABOVE this
  export HASHIRU_SEMANTIC_DENSITY_THRESHOLD=0.8 # reprompt concern if density BELOW this

Values for enable flags: 1/0, true/false, yes/no, on/off (case-insensitive).
Empty env = default (enabled for toggles; numeric defaults for thresholds).

Stochastic samples per AskAgent: ``HASHIRU_SEMANTIC_EXTRA_SAMPLES`` (default 4, max 16).
Long-CoT + HTTP logprobs: ``HASHIRU_OLLAMA_SKIP_HTTP_LOGPROBS_REASONING`` in ``ollama_logprobs.py``.
"""
from __future__ import annotations

import os


def _truthy(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    v = str(raw).strip().lower()
    if v in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    if v in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    return default


SEMANTIC_ENTROPY_THRESHOLD = float(
    os.environ.get("HASHIRU_SEMANTIC_ENTROPY_THRESHOLD", "1.65")
)
SEMANTIC_DENSITY_THRESHOLD = float(
    os.environ.get("HASHIRU_SEMANTIC_DENSITY_THRESHOLD", "0.8")
)


def semantic_entropy_enabled() -> bool:
    return _truthy("HASHIRU_ENABLE_SEMANTIC_ENTROPY", default=True)


def semantic_density_enabled() -> bool:
    return _truthy("HASHIRU_ENABLE_SEMANTIC_DENSITY", default=True)


def ask_multiple_agents_tool_enabled() -> bool:
    """Expose and allow the AskMultipleAgents CEO tool (default: enabled)."""
    return _truthy("HASHIRU_ENABLE_ASK_MULTIPLE_AGENTS", default=True)


def semantic_metrics_sampling_enabled() -> bool:
    """Extra stochastic samples + gateway calls only if at least one metric is enabled."""
    return semantic_entropy_enabled() or semantic_density_enabled()


def apply_ablation_to_metrics(entropy, density):
    """
    After compute_semantic_metrics_both, null out disabled branches for history / CEO.
    """
    e_on = semantic_entropy_enabled()
    d_on = semantic_density_enabled()
    return (
        entropy if e_on else None,
        density if d_on else None,
    )


def reprompt_triggered(entropy, density) -> bool:
    """
    True when any *enabled* metric crosses its threshold (high entropy or low density).
    Used for semantic_quality_concern / CEO guidance; workers are not auto re-prompted.
    """
    e_on = semantic_entropy_enabled()
    d_on = semantic_density_enabled()
    if not e_on and not d_on:
        return False
    if e_on and entropy is not None and entropy > SEMANTIC_ENTROPY_THRESHOLD:
        return True
    if d_on and density is not None and density < SEMANTIC_DENSITY_THRESHOLD:
        return True
    return False


def flags_dict() -> dict:
    return {
        "semantic_entropy_enabled": semantic_entropy_enabled(),
        "semantic_density_enabled": semantic_density_enabled(),
        "semantic_entropy_threshold": SEMANTIC_ENTROPY_THRESHOLD,
        "semantic_density_threshold": SEMANTIC_DENSITY_THRESHOLD,
    }


def reprompt_explanation_fragment(entropy, density) -> str:
    """Human-readable metric snippet (logging / diagnostics)."""
    parts = []
    if semantic_entropy_enabled():
        parts.append(
            f"semantic_entropy={entropy:.3f}" if entropy is not None else "semantic_entropy=n/a"
        )
    if semantic_density_enabled():
        parts.append(
            f"semantic_density={density:.3f}" if density is not None else "semantic_density=n/a"
        )
    return ", ".join(parts) if parts else "semantic metrics off"


def semantic_threshold_violations(entropy, density) -> list[str]:
    """Short machine-readable violation tags when enabled metrics cross thresholds."""
    out: list[str] = []
    if semantic_entropy_enabled() and entropy is not None and entropy > SEMANTIC_ENTROPY_THRESHOLD:
        out.append(
            f"entropy_too_high:{float(entropy):.4f}>{SEMANTIC_ENTROPY_THRESHOLD}"
        )
    if semantic_density_enabled() and density is not None and density < SEMANTIC_DENSITY_THRESHOLD:
        out.append(
            f"density_too_low:{float(density):.4f}<{SEMANTIC_DENSITY_THRESHOLD}"
        )
    return out


def ceo_semantic_quality_followup(
    *,
    agent_name: str,
    worker_prompt: str,
    entropy,
    density,
) -> dict:
    """
    Structured guidance for the CEO when semantic checks fail after the first worker answer.
    """
    violations = semantic_threshold_violations(entropy, density)
    concern = len(violations) > 0
    if not concern:
        return {
            "semantic_quality_concern": False,
            "semantic_threshold_violations": [],
            "semantic_quality_summary": "",
            "suggested_ceo_next_steps": [],
            "worker_prompt_excerpt_for_ceo": "",
        }
    wp = worker_prompt.strip() if isinstance(worker_prompt, str) else ""
    excerpt = (wp[:240] + "…") if len(wp) > 240 else wp
    summary = (
        f"Worker `{agent_name}` returned a first-pass answer, but semantic quality crossed thresholds "
        f"({'; '.join(violations)}). No worker auto-reprompt ran—you must decide the next action."
    )
    steps = [
        "Reprompt the same agent with clearer instructions or ask for explicit uncertainty handling.",
    ]
    if ask_multiple_agents_tool_enabled():
        steps.append(
            "Use AskMultipleAgents with focused sub-prompts, then synthesize one answer."
        )
    else:
        steps.append(
            "Use AskAgent again with a different agent or a tighter sub-prompt (AskMultipleAgents is disabled)."
        )
    steps.extend(
        [
            "Delegate to a different agent or a better-fit base model.",
            "If this agent repeatedly crosses thresholds, retire it and recreate with a sharper system prompt.",
        ]
    )
    return {
        "semantic_quality_concern": True,
        "semantic_threshold_violations": violations,
        "semantic_quality_summary": summary,
        "suggested_ceo_next_steps": steps,
        "worker_prompt_excerpt_for_ceo": excerpt,
    }
