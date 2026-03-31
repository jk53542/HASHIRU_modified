"""
Restrict which worker **base_model** values HASHIRU may load, create, or advertise.

Ablations / SOTA runs:
  export HASHIRU_WORKER_MODEL_FAMILY=groq          # only Groq-hosted workers
  export HASHIRU_WORKER_MODEL_FAMILY=deepseek    # only Ollama deepseek-r1
  export HASHIRU_WORKER_MODEL_FAMILY=llama       # only Ollama llama3.2

Exact override (comma-separated base_model ids as in AgentCostManager):
  export HASHIRU_WORKER_BASE_MODELS=deepseek-r1,groq-llama-3.3-70b-versatile

Unset both variables to allow all models listed in AgentCostManager.

Families (case-insensitive):
  deepseek, llama, mistral  — single Ollama-backed base
  ollama, local             — all Ollama worker bases (deepseek-r1, llama3.2, mistral)
  groq                      — Groq cloud workers
  lambda                    — Lambda Labs workers
  openai, chatgpt           — OpenAI-hosted ChatGPT workers
  gemini                    — Gemini workers (requires enabling gemini in AgentManager)
"""
from __future__ import annotations

import os
from typing import FrozenSet, Optional

# Must stay in sync with keys in src/tools/default_tools/agent_cost_manager.py
_OLLAMA_BASES = frozenset({"deepseek-r1", "llama3.2", "mistral"})
_GROQ_BASES = frozenset({"groq-llama-3.3-70b-versatile"})
_LAMBDA_BASES = frozenset({"lambda-hermes3-8b"})
_OPENAI_BASES = frozenset({"chatgpt-5.4"})
_GEMINI_BASES = frozenset(
    {
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    }
)

_FAMILY_ALIASES: dict[str, FrozenSet[str]] = {
    "deepseek": frozenset({"deepseek-r1"}),
    "llama": frozenset({"llama3.2"}),
    "mistral": frozenset({"mistral"}),
    "ollama": _OLLAMA_BASES,
    "local": _OLLAMA_BASES,
    "groq": _GROQ_BASES,
    "lambda": _LAMBDA_BASES,
    "openai": _OPENAI_BASES,
    "chatgpt": _OPENAI_BASES,
    "gemini": _GEMINI_BASES,
}


def _normalize_family(raw: str) -> str:
    return (raw or "").strip().lower()


def get_effective_allowlist() -> Optional[FrozenSet[str]]:
    """
    None = no restriction (all AgentCostManager models allowed).
    """
    explicit = os.environ.get("HASHIRU_WORKER_BASE_MODELS", "").strip()
    if explicit:
        parts = {p.strip() for p in explicit.split(",") if p.strip()}
        return frozenset(parts) if parts else None

    fam = _normalize_family(os.environ.get("HASHIRU_WORKER_MODEL_FAMILY", ""))
    if not fam:
        return None
    if fam not in _FAMILY_ALIASES:
        allowed = ", ".join(sorted(_FAMILY_ALIASES.keys()))
        raise ValueError(
            f"Unknown HASHIRU_WORKER_MODEL_FAMILY={fam!r}. "
            f"Use one of: {allowed}"
        )
    return _FAMILY_ALIASES[fam]


def is_base_model_allowed(base_model: str) -> bool:
    allow = get_effective_allowlist()
    if allow is None:
        return True
    return base_model in allow


def assert_base_model_allowed(base_model: str) -> None:
    if not is_base_model_allowed(base_model):
        allow = get_effective_allowlist()
        raise ValueError(
            f"base_model {base_model!r} is not allowed under the active worker model policy. "
            f"Allowed: {sorted(allow) if allow else []}. "
            f"Adjust HASHIRU_WORKER_MODEL_FAMILY or HASHIRU_WORKER_BASE_MODELS, or unset both."
        )


def policy_summary() -> dict:
    allow = get_effective_allowlist()
    return {
        "restricted": allow is not None,
        "allowed_base_models": sorted(allow) if allow else None,
        "family_env": os.environ.get("HASHIRU_WORKER_MODEL_FAMILY"),
        "explicit_models_env": os.environ.get("HASHIRU_WORKER_BASE_MODELS"),
    }
