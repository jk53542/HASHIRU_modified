"""
Append-only JSONL trace for CEO / worker orchestration (ablations, interpretability).

Enable:
  export HASHIRU_ORCHESTRATION_TRACE_JSONL=/path/to/run.jsonl
  export HASHIRU_ORCHESTRATION_TRACE_JSONL=/path/to/logs_dir

Path rules:

- Ends with ``.jsonl`` → **single file**; every process appends to that path (can grow large).
- Anything else → treated as a **directory**; each HASHIRU process gets its own file:
  ``trace_<YYYYMMDD_HHMMSS>_<pid>.jsonl``. Start a new file by restarting HASHIRU (new process).

If HASHIRU runs under **WSL**, set variables in the **same WSL shell** before ``python app.py``,
with Linux paths. ``export VAR="~/x"`` does not expand ``~`` in bash — use ``VAR=~/x`` or absolute paths.

Each line is one JSON object with at least: ts (unix), event (str), plus fields. Event lines also
include ``trace_file`` (basename) in directory mode.

When a user message starts with ``[HASHIRU_TRACE_CTX]{json}\\n`` (see benchmarks using
``benchmark_trace_context.hashiru_trace_context_prefix``), the server strips that line and attaches
``benchmark_name``, ``question_index``, ``question_id``, ``bench_attempt``, and optional
``question_text`` to every trace row for that CEO turn. The visible user message (after stripping
the prefix) is also stored as ``user_turn_excerpt`` so downstream tools can align traces to
benchmark rows even when worker prompts paraphrase the question.

Turn metadata is stored both in a ``ContextVar`` and a thread-local fallback so tool logging
still sees the same turn if the runtime moves work across contexts.

``worker_reprompted_after_semantic_check`` is True on a worker completion when an earlier completion
for the **same agent name** in the same user turn had ``semantic_quality_concern`` true (CEO chose
to ask again after thresholds fired). ``worker_invocation_index`` counts that agent's completions in
the turn (1-based). That distinguishes **semantic reprompts** (same routing, new ``worker_prompt``)
from **AskMultipleAgents** (``worker_answer_multi`` / ``worker_routing: AskMultipleAgents``).

**Worker round cap:** each completed ``AskAgent`` or ``AskMultipleAgents`` tool call counts as one
worker tool round for the current user message. After ``1 + HASHIRU_MAX_CEO_WORKER_REPROMPTS``
rounds (default 5 reprompts → 6 rounds), further worker tools return a cap message and a heuristic
best prior output instead of invoking models. See ``src/manager/ceo_worker_round_policy.py``.

Each ``worker_answer`` / ``worker_answer_multi`` includes ``worker_response`` (the **primary**
completion only—the one returned to the CEO for that tool call). Entropy/density may use
additional **same-prompt** stochastic completions; those are **not** logged as separate trace rows.
Optional: set ``HASHIRU_TRACE_SEMANTIC_AUXILIARY_RESPONSES=1`` to also record the texts of those
auxiliary samples in fields ``semantic_auxiliary_responses`` (list of strings, each truncated in the
JSONL writer). Default is **off** to keep traces small.
The text the CEO sees for that ``AskAgent`` call is **always** the first completion (index **0**):
``same_prompt_completion_index_sent_to_ceo`` is always ``0`` and ``worker_response`` matches the
tool output. ``same_prompt_total_llm_completions`` is ``1 + semantic_auxiliary_completions_count``.
See also ``worker_response_kind`` (``primary``) and ``semantic_metrics_include_auxiliary_samples``.
Benchmark turns should set ``question_text`` via ``[HASHIRU_TRACE_CTX]`` so every line also carries
the original benchmark question.

Each line also carries ``semantic_entropy`` / ``semantic_density``, threshold fields, and
``implied_ceo_decision_hint`` summarizing whether the CEO *should* re-evaluate or may proceed.

``ceo_final_answer`` is emitted once per completed CEO turn when ``invoke_manager`` stops recursing
(no further tool rounds in that branch). ``ceo_final_answer`` is the same string benchmarks use as
``agent_final_response`` (last assistant ``content`` in history, matching
``get_last_assistant_content`` in jailbreakbench). It is present even when the model only returned
tool calls in the last stream chunk (then the logged text may be an earlier assistant message or
empty if history ends on tool/user turns).

**CEO token fields** on ``ceo_final_answer`` (from ``GeminiManager`` counters): ``ceo_session_input_tokens`` /
``ceo_session_output_tokens`` are cumulative for the process since manager construction. When the UI
calls ``begin_orchestration_user_turn`` with ``ceo_input_tokens_baseline`` / ``ceo_output_tokens_baseline``
(Gradio does this before each user message), the same row also includes ``ceo_turn_delta_input_tokens`` /
``ceo_turn_delta_output_tokens`` for that question/turn. **Input** totals are dominated by Gemini
``count_tokens`` on each CEO ``generate_content`` request; **output** totals use the same heuristics
as budgeting (word-split counts on streamed text and ``repr`` splits on tool payloads)—not worker
model usage tokens.

On ``ceo_tool_finished`` for worker tools: ``worker_routing`` is ``AskAgent`` (one worker per tool
call) or ``AskMultipleAgents`` (several workers in one tool call). ``worker_subcalls_this_tool`` is
the number of worker completions in that tool call (1 for AskAgent; for AskMultipleAgents, the
length of ``subcall_agent_names``, which lists one name per subcall and may repeat the same agent).
``called_agent_names`` remains de-duplicated for quick scanning. ``worker_response`` is duplicated
on ``ceo_tool_finished`` for ``AskAgent`` when available.

**CEO decision trace (``ceo_decision``):** logged immediately *before* each delegation tool
(``AskAgent``, ``AskMultipleAgents``, ``AgentCreator``, ``FireAgent``) with required rationale
fields from the tool call (``ceo_rationale``, ``agent_selection_rationale``,
``labor_division_rationale``, etc.) plus ``prior_semantic_context`` summarizing recent worker
entropy/density concerns. ``ceo_final_answer`` may include ``ceo_synthesis_rationale`` when the CEO
prefixes it with ``CEO_SYNTHESIS_RATIONALE:`` and ``ceo_decision_history`` for the turn.

Call ``init_orchestration_trace_session()`` once at app startup (optional but recommended) to emit
``trace_session_start`` and create the per-session file immediately.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from src.manager.ceo_worker_round_policy import (
    max_ceo_worker_reprompts_per_user_turn,
    max_ceo_worker_tool_rounds_per_user_turn,
)

# Per user-message turn: benchmark labels + per-agent semantic-concern history (for reprompt detection).
_orch_turn: ContextVar[dict[str, Any] | None] = ContextVar("hashiru_orch_turn", default=None)
# Mirror turn dict when ContextVar is empty in this thread (e.g. some Gradio/async boundaries).
_orch_turn_tls = threading.local()

_TRACE_CTX_PREFIX = "[HASHIRU_TRACE_CTX]"
_ALLOWED_TRACE_CTX_KEYS = frozenset(
    {
        "benchmark_name",
        "question_index",
        "question_id",
        "bench_attempt",
        "question_text",
    }
)
_logger = logging.getLogger(__name__)

# When tracing auxiliary semantic samples as text, cap list length and per-string size in JSONL.
_TRACE_AUX_RESPONSES_MAX_ITEMS = 32
_TRACE_AUX_RESPONSES_STR_LIMIT = 12000

_session_lock = threading.Lock()
# Resolved absolute path to the active JSONL file; None = not yet resolved; "" = tracing disabled
_trace_abs_path: str | None = None
_trace_basename: str = ""
_session_header_written: bool = False


def trace_path() -> str | None:
    raw = os.environ.get("HASHIRU_ORCHESTRATION_TRACE_JSONL", "").strip()
    if not raw:
        return None
    p = os.path.normpath(os.path.expandvars(os.path.expanduser(raw)))
    return p or None


def trace_include_semantic_auxiliary_responses() -> bool:
    """
    When true, worker trace events may include ``semantic_auxiliary_responses`` (texts of
    same-prompt stochastic samples used for entropy/density). Default false.
    """
    v = os.environ.get("HASHIRU_TRACE_SEMANTIC_AUXILIARY_RESPONSES", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _explicit_jsonl_file(configured: str) -> bool:
    return configured.lower().endswith(".jsonl")


def _session_filename() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"trace_{stamp}_{os.getpid()}.jsonl"


def _ensure_trace_target_unlocked(configured: str) -> str | None:
    """
    Set global trace file path on first use. Caller must hold _session_lock.
    Returns absolute path or None if tracing unavailable.
    """
    global _trace_abs_path, _trace_basename

    if _trace_abs_path is not None:
        if _trace_abs_path == "":
            return None
        return _trace_abs_path

    try:
        if _explicit_jsonl_file(configured):
            abs_path = os.path.abspath(configured)
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            _trace_basename = os.path.basename(abs_path)
            _trace_abs_path = abs_path
            return abs_path

        dir_abs = os.path.abspath(configured)
        os.makedirs(dir_abs, exist_ok=True)
        fname = _session_filename()
        abs_path = os.path.join(dir_abs, fname)
        _trace_basename = fname
        _trace_abs_path = abs_path
        return abs_path
    except OSError as e:
        _logger.warning(
            "HASHIRU_ORCHESTRATION_TRACE_JSONL: could not init (%s): %s",
            configured,
            e,
        )
        _trace_abs_path = ""
        return None


def init_orchestration_trace_session() -> str | None:
    """
    Create the session trace file and write a single ``trace_session_start`` line.
    Safe to call once at app startup; no-op if tracing is disabled or header already written.
    """
    global _session_header_written

    cfg = trace_path()
    if not cfg:
        with _session_lock:
            global _trace_abs_path
            if _trace_abs_path is None:
                _trace_abs_path = ""
        return None

    with _session_lock:
        if _session_header_written:
            return _trace_abs_path if _trace_abs_path not in (None, "") else None
        abs_path = _ensure_trace_target_unlocked(cfg)
        if not abs_path:
            return None
        try:
            meta = {
                "ts": time.time(),
                "event": "trace_session_start",
                "trace_file": _trace_basename,
                "trace_path": abs_path,
                "pid": os.getpid(),
            }
            line = json.dumps(meta, default=str, ensure_ascii=False) + "\n"
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(line)
            _session_header_written = True
        except OSError as e:
            _logger.warning(
                "HASHIRU_ORCHESTRATION_TRACE_JSONL: session start write failed (%s): %s",
                abs_path,
                e,
            )
            return None
    return abs_path


def _default_turn_state() -> dict[str, Any]:
    return {
        "benchmark_name": None,
        "question_index": None,
        "question_id": None,
        "bench_attempt": None,
        "question_text": None,
        "user_turn_excerpt": None,
        "concern_by_agent": {},
        # AskAgent + AskMultipleAgents completions this user turn (see ceo_worker_round_policy).
        "ceo_worker_tool_rounds_completed": 0,
        "ceo_worker_round_history": [],
        "ceo_decision_history": [],
        # GeminiManager counters at turn start (optional; enables per-question deltas on ceo_final_answer).
        "ceo_input_tokens_baseline": None,
        "ceo_output_tokens_baseline": None,
    }


def _get_turn_state() -> dict[str, Any] | None:
    st = _orch_turn.get()
    if st is not None:
        return st
    return getattr(_orch_turn_tls, "turn", None)


def begin_orchestration_user_turn(
    meta: dict[str, Any] | None = None,
    *,
    user_turn_excerpt: str | None = None,
    ceo_input_tokens_baseline: int | None = None,
    ceo_output_tokens_baseline: int | None = None,
) -> None:
    """
    Reset orchestration state for one CEO user turn (one Gradio user message).
    Optionally set benchmark_* fields copied into every trace line until the next turn.
    Pass ``user_turn_excerpt`` with the CEO-visible user text (prefix already stripped).

    Pass ``ceo_*_tokens_baseline`` from ``GeminiManager.input_tokens`` / ``output_tokens`` immediately
    before handling the user message so ``ceo_final_answer`` rows can record per-turn deltas.
    """
    st = _default_turn_state()
    if meta:
        for k in _ALLOWED_TRACE_CTX_KEYS:
            if k in meta and meta[k] is not None:
                st[k] = meta[k]
    if st.get("question_index") is not None:
        try:
            st["question_index"] = int(st["question_index"])
        except (TypeError, ValueError):
            pass
    if user_turn_excerpt:
        st["user_turn_excerpt"] = user_turn_excerpt.strip()[:8000]
    if ceo_input_tokens_baseline is not None:
        try:
            st["ceo_input_tokens_baseline"] = int(ceo_input_tokens_baseline)
        except (TypeError, ValueError):
            pass
    if ceo_output_tokens_baseline is not None:
        try:
            st["ceo_output_tokens_baseline"] = int(ceo_output_tokens_baseline)
        except (TypeError, ValueError):
            pass
    _orch_turn.set(st)
    _orch_turn_tls.turn = st


def parse_trace_context_from_user_text(text: str) -> tuple[str, dict[str, Any] | None]:
    """
    If ``text`` starts with ``[HASHIRU_TRACE_CTX]{json}`` (JSON on the first line),
    strip it and return (remaining_text, meta). Otherwise return (text, None).

    Benchmarks prepend this so JSONL traces can be grouped by question.
    """
    if not text or not text.startswith(_TRACE_CTX_PREFIX):
        return text, None
    rest = text[len(_TRACE_CTX_PREFIX) :]
    if "\n" in rest:
        line, remainder = rest.split("\n", 1)
    else:
        line, remainder = rest, ""
    line = line.strip()
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return text, None
    if not isinstance(obj, dict):
        return text, None
    meta = {k: obj[k] for k in _ALLOWED_TRACE_CTX_KEYS if k in obj}
    qt = meta.get("question_text")
    if isinstance(qt, str) and len(qt) > 2500:
        meta["question_text"] = qt[:2500]
    clean = remainder.lstrip("\n")
    return clean, meta


def _turn_context_trace_fields() -> dict[str, Any]:
    st = _get_turn_state()
    if not st:
        return {}
    out: dict[str, Any] = {}
    for k in _ALLOWED_TRACE_CTX_KEYS:
        v = st.get(k)
        if v is not None:
            out[k] = v
    ex = st.get("user_turn_excerpt")
    if ex:
        out["user_turn_excerpt"] = ex
    return out


def ceo_token_fields_for_final_answer(
    session_input_tokens: int,
    session_output_tokens: int,
) -> dict[str, Any]:
    """
    Fields appended to ``ceo_final_answer`` trace rows. Session totals are always emitted;
    per-turn deltas appear when ``begin_orchestration_user_turn`` recorded baselines.
    """
    out: dict[str, Any] = {
        "ceo_session_input_tokens": int(session_input_tokens),
        "ceo_session_output_tokens": int(session_output_tokens),
    }
    st = _get_turn_state()
    if not st:
        return out
    bi = st.get("ceo_input_tokens_baseline")
    bo = st.get("ceo_output_tokens_baseline")
    if bi is not None:
        out["ceo_turn_delta_input_tokens"] = max(0, int(session_input_tokens) - int(bi))
    if bo is not None:
        out["ceo_turn_delta_output_tokens"] = max(0, int(session_output_tokens) - int(bo))
    if bi is not None and bo is not None:
        out["ceo_turn_delta_total_tokens"] = out["ceo_turn_delta_input_tokens"] + out[
            "ceo_turn_delta_output_tokens"
        ]
    return out


def worker_invocation_reprompt_flags(
    agent_name: str, semantic_quality_concern: bool
) -> tuple[int, bool]:
    """
    Within the current user turn, track each worker's sequence of semantic_quality_concern outcomes.

    Returns (worker_invocation_index_1based, worker_reprompted_after_semantic_check).
    The reprompt flag is True when a prior completion for this agent in the same turn
    had semantic_quality_concern True (CEO likely re-asked after thresholds fired).
    """
    st = _get_turn_state()
    if not st:
        return 1, False
    name = (agent_name or "").strip() or "_unknown"
    chains: dict[str, list[bool]] = st.setdefault("concern_by_agent", {})
    hist = chains.setdefault(name, [])
    invocation_index = len(hist) + 1
    reprompted = len(hist) >= 1 and bool(hist[-1])
    hist.append(bool(semantic_quality_concern))
    return invocation_index, reprompted


def _ceo_worker_rounds_completed(st: dict[str, Any]) -> int:
    try:
        return int(st.get("ceo_worker_tool_rounds_completed") or 0)
    except (TypeError, ValueError):
        return 0


def ceo_worker_tool_round_cap_reached() -> bool:
    """
    True when the current user turn has already used the allowed number of
    AskAgent / AskMultipleAgents tool rounds (initial + reprompts).
    """
    st = _get_turn_state()
    if not st:
        return False
    return _ceo_worker_rounds_completed(st) >= max_ceo_worker_tool_rounds_per_user_turn()


def record_ceo_worker_tool_round(summary: dict[str, Any]) -> None:
    """Append one completed worker delegation round for cap tracking and best-of selection."""
    st = _get_turn_state()
    if not st:
        return
    st["ceo_worker_tool_rounds_completed"] = _ceo_worker_rounds_completed(st) + 1
    st.setdefault("ceo_worker_round_history", []).append(summary)


def _score_round_for_best(s: dict[str, Any]) -> tuple[float, float, float]:
    """
    Lower is better: prefer no semantic concern, higher density, lower entropy.
    Non-finite metrics sort last.
    """
    concern = 1.0 if s.get("semantic_quality_concern") else 0.0
    dens = s.get("semantic_density")
    ent = s.get("semantic_entropy")
    try:
        d = float(dens)
        if d != d:  # NaN
            d = -1.0
    except (TypeError, ValueError):
        d = -1.0
    try:
        e = float(ent)
        if e != e:
            e = 1e100
    except (TypeError, ValueError):
        e = 1e100
    return (concern, -d, e)


def select_best_prior_worker_round(history: list[Any]) -> dict[str, Any] | None:
    """Pick a single prior round using a small semantic-metrics heuristic."""
    rounds = [x for x in history if isinstance(x, dict)]
    if not rounds:
        return None
    return min(rounds, key=_score_round_for_best)


def _digest_worker_round_history(history: list[Any], preview_len: int = 400) -> list[dict[str, Any]]:
    """Short summaries for tool JSON (full text already appears in prior chat tool results)."""
    out: list[dict[str, Any]] = []
    for i, h in enumerate(history):
        if not isinstance(h, dict):
            continue
        blob = (
            (h.get("primary_output_excerpt") or "")
            or (h.get("worker_response") or "")
            or (h.get("combined_output") or "")
        )
        blob = str(blob)
        prev = blob[:preview_len] + ("…" if len(blob) > preview_len else "")
        one: dict[str, Any] = {
            "round_index": i + 1,
            "tool": h.get("tool"),
            "semantic_quality_concern": h.get("semantic_quality_concern"),
            "semantic_density": h.get("semantic_density"),
            "semantic_entropy": h.get("semantic_entropy"),
            "output_preview": prev,
        }
        if h.get("tool") == "AskAgent":
            one["agent_name"] = h.get("agent_name")
        elif h.get("tool") == "AskMultipleAgents":
            names = []
            for row in h.get("per_agent_outputs_brief") or []:
                if isinstance(row, dict) and row.get("agent_name"):
                    names.append(row["agent_name"])
            if names:
                one["agent_names"] = names
        out.append(one)
    return out


def _slim_round_for_ceo(h: dict[str, Any] | None) -> dict[str, Any] | None:
    if not h:
        return None
    keys = (
        "tool",
        "agent_name",
        "user_question",
        "semantic_quality_concern",
        "semantic_entropy",
        "semantic_density",
        "semantic_metric_scope",
        "primary_output_excerpt",
        "worker_prompt",
        "worker_response",
        "combined_output",
    )
    slim = {k: h.get(k) for k in keys if k in h and h.get(k) is not None}
    if h.get("tool") == "AskMultipleAgents" and h.get("per_agent_outputs_brief"):
        slim["per_agent_outputs_brief"] = h.get("per_agent_outputs_brief")
    return slim


def ceo_worker_round_cap_tool_result(*, requested_tool: str) -> dict[str, Any]:
    """
    Tool return value when the CEO hits the per-turn worker delegation cap.
    The CEO must not call AskAgent / AskMultipleAgents again for this user message.
    """
    st = _get_turn_state()
    history: list = list((st or {}).get("ceo_worker_round_history") or [])
    best = select_best_prior_worker_round(history)
    best_slim = _slim_round_for_ceo(best)
    cap = max_ceo_worker_tool_rounds_per_user_turn()
    n_done = len(history)
    max_rep = max_ceo_worker_reprompts_per_user_turn()

    best_text = ""
    if best:
        best_text = (
            (best.get("primary_output_excerpt") or "")
            or (best.get("worker_response") or "")
            or (best.get("combined_output") or "")
        )

    msg = (
        f"Worker delegation cap reached for this user message: completed {n_done} worker tool "
        f"round(s) (maximum {cap}, i.e. {max_rep} reprompt(s) after the first). "
        f"Do NOT call AskAgent or AskMultipleAgents again for this turn. "
        f"Produce your final answer to the user by synthesizing from prior worker output(s). "
        f"The field heuristic_best_prior_round summarizes one reasonable choice; you may override "
        f"with your own judgment."
    )

    orch_meta: dict[str, Any] = {
        "agent_name": None,
        "worker_prompt": f"(blocked: {requested_tool} — worker round cap reached)",
        "worker_response": best_text,
        "worker_response_kind": "primary",
        "semantic_auxiliary_completions_count": 0,
        "semantic_metrics_include_auxiliary_samples": False,
        "same_prompt_completion_index_sent_to_ceo": 0,
        "same_prompt_total_llm_completions": 1,
        "semantic_entropy": (best or {}).get("semantic_entropy"),
        "semantic_density": (best or {}).get("semantic_density"),
        "semantic_entropy_threshold": None,
        "semantic_density_threshold": None,
        "semantic_quality_concern": False,
        "worker_invocation_index": None,
        "worker_reprompted_after_semantic_check": False,
        "semantic_threshold_violations": [],
        "semantic_quality_summary": "Worker round cap; no new worker call executed.",
        "suggested_ceo_next_steps": [
            "Synthesize the user-facing reply from prior worker outputs without further delegation.",
        ],
        "implied_ceo_decision_hint": "worker_round_cap_reached_finalize_from_prior_outputs",
        "worker_round_cap_reached": True,
    }

    out: dict[str, Any] = {
        "status": "success",
        "message": msg,
        "worker_round_cap_reached": True,
        "worker_rounds_completed": n_done,
        "max_worker_rounds_allowed": cap,
        "max_reprompts_allowed": max_rep,
        "prior_worker_rounds_digest": _digest_worker_round_history(history),
        "heuristic_best_prior_round": best_slim,
        "semantic_quality_concern": False,
        "semantic_threshold_violations": [],
        "semantic_quality_summary": orch_meta["semantic_quality_summary"],
        "suggested_ceo_next_steps": orch_meta["suggested_ceo_next_steps"],
        "worker_prompt_excerpt_for_ceo": orch_meta["worker_prompt"],
        "orchestration_meta": orch_meta,
    }

    if requested_tool == "AskAgent":
        out["output"] = best_text if best_text else None
    else:
        out["output"] = best_text if best_text else None
        out["combined_output"] = (
            (best or {}).get("combined_output") or best_text or ""
        )
        pa = (best or {}).get("per_agent_outputs_brief")
        out["per_agent_outputs"] = pa if isinstance(pa, list) else []
        out["semantic_entropy"] = (best or {}).get("semantic_entropy")
        out["semantic_density"] = (best or {}).get("semantic_density")
        out["semantic_metric_scope"] = (best or {}).get("semantic_metric_scope")

    return out


def _truncate(val: Any, limit: int = 4000, nested_str_limit: int = 4000) -> Any:
    if isinstance(val, str) and len(val) > limit:
        return val[:limit] + f"...<truncated {len(val) - limit} chars>"
    if isinstance(val, dict):
        return {
            k: _truncate(v, limit=nested_str_limit, nested_str_limit=min(nested_str_limit, 8000))
            for k, v in list(val.items())[:80]
        }
    if isinstance(val, (list, tuple)) and len(val) > 64:
        return [
            _truncate(x, limit=min(limit, nested_str_limit), nested_str_limit=nested_str_limit)
            for x in val[:10]
        ] + [f"...<{len(val) - 10} more>"]
    return val


def _truncate_per_agent_outputs(rows: Any, per_text_limit: int = 16000) -> Any:
    """Keep multi-agent trace readable: cap each prompt/response string."""
    if not isinstance(rows, list):
        return _truncate(rows)
    slim: list[dict[str, Any]] = []
    for item in rows[:48]:
        if not isinstance(item, dict):
            continue
        one: dict[str, Any] = {}
        for k, v in item.items():
            if k in ("response", "prompt", "worker_response") and isinstance(v, str):
                one[k] = _truncate(v, limit=per_text_limit)
            else:
                one[k] = _truncate(v, limit=4000, nested_str_limit=4000)
        slim.append(one)
    if len(rows) > 48:
        slim.append({"_note": f"...<{len(rows) - 48} more agents omitted>"})
    return slim


def log_orchestration_event(event: str, **fields: Any) -> None:
    global _session_header_written

    cfg = trace_path()
    if not cfg:
        return

    with _session_lock:
        abs_path = _ensure_trace_target_unlocked(cfg)
        if not abs_path:
            return

        if not _session_header_written:
            try:
                if (not os.path.isfile(abs_path)) or os.path.getsize(abs_path) == 0:
                    meta = {
                        "ts": time.time(),
                        "event": "trace_session_start",
                        "trace_file": _trace_basename,
                        "trace_path": abs_path,
                        "pid": os.getpid(),
                    }
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(meta, default=str, ensure_ascii=False) + "\n")
            except OSError as e:
                _logger.warning(
                    "HASHIRU_ORCHESTRATION_TRACE_JSONL: lazy session header failed: %s", e
                )
            _session_header_written = True

        rec: dict[str, Any] = {"ts": time.time(), "event": event}
        if _trace_basename:
            rec["trace_file"] = _trace_basename
        rec.update(_turn_context_trace_fields())
        for k, v in fields.items():
            if k == "worker_response" and isinstance(v, str):
                rec[k] = _truncate(v, limit=50000)
            elif k == "ceo_final_answer" and isinstance(v, str):
                rec[k] = _truncate(v, limit=50000)
            elif k in (
                "ceo_rationale",
                "agent_selection_rationale",
                "labor_division_rationale",
                "creation_rationale",
                "retirement_rationale",
                "semantic_metrics_influence",
                "ceo_synthesis_rationale",
                "last_round_semantic_quality_summary",
            ) and isinstance(v, str):
                rec[k] = _truncate(v, limit=16000)
            elif k == "ceo_decision_history" and isinstance(v, list):
                rec[k] = [
                    _truncate(x, limit=4000, nested_str_limit=16000)
                    if isinstance(x, dict)
                    else _truncate(x, limit=4000)
                    for x in v[:24]
                ]
            elif k == "prior_semantic_context" and isinstance(v, dict):
                rec[k] = _truncate(v, limit=4000, nested_str_limit=16000)
            elif k == "semantic_auxiliary_responses" and isinstance(v, list):
                rec[k] = [
                    _truncate(str(x), limit=_TRACE_AUX_RESPONSES_STR_LIMIT)
                    for x in v[:_TRACE_AUX_RESPONSES_MAX_ITEMS]
                ]
                if len(v) > _TRACE_AUX_RESPONSES_MAX_ITEMS:
                    rec[k].append(
                        f"...<{len(v) - _TRACE_AUX_RESPONSES_MAX_ITEMS} more samples omitted>"
                    )
            elif k == "per_agent_outputs":
                rec[k] = _truncate_per_agent_outputs(v)
            else:
                rec[k] = _truncate(v)
        line = json.dumps(rec, default=str, ensure_ascii=False) + "\n"
        try:
            with open(abs_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            _logger.warning(
                "HASHIRU_ORCHESTRATION_TRACE_JSONL: could not append (%s): %s",
                abs_path,
                e,
            )
