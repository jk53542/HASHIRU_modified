"""
CEO decision justification fields for orchestration JSONL traces.

Delegation tools (AskAgent, AskMultipleAgents, AgentCreator, FireAgent) require the
CEO to pass explicit rationale strings so traces expose routing mistakes—especially
when semantic entropy/density influenced the next action.
"""
from __future__ import annotations

from typing import Any

from src.manager.orchestration_trace import (
    _get_turn_state,
    log_orchestration_event,
    select_best_prior_worker_round,
)

CEO_DELEGATION_TOOLS = frozenset(
    {"AskAgent", "AskMultipleAgents", "AgentCreator", "FireAgent"}
)

CEO_RATIONALE_FIELD_NAMES = frozenset(
    {
        "ceo_rationale",
        "agent_selection_rationale",
        "labor_division_rationale",
        "creation_rationale",
        "retirement_rationale",
        "semantic_metrics_influence",
        "ceo_synthesis_rationale",
    }
)

_CEO_RATIONALE_STR_LIMIT = 16000

_CEO_RATIONALE_BASE_PROPERTIES: dict[str, dict[str, str]] = {
    "ceo_rationale": {
        "type": "string",
        "description": (
            "Brief justification for this action now: why this tool, why this timing, "
            "and how it fits your decomposition plan for the user question."
        ),
    },
    "semantic_metrics_influence": {
        "type": "string",
        "description": (
            "If a prior worker reply had semantic_quality_concern (entropy/density thresholds), "
            "explain whether and how those metrics influenced this decision. "
            "If this is the first worker call or metrics did not influence you, state that explicitly "
            "(e.g. 'N/A - first worker call' or 'N/A - proceeding despite prior concern because ...')."
        ),
    },
}

_AGENT_SELECTION_PROPERTY: dict[str, dict[str, str]] = {
    "agent_selection_rationale": {
        "type": "string",
        "description": (
            "Why these agent(s) were chosen: specialty fit, reuse vs newly created agent, "
            "and why this base model is appropriate for the subtask."
        ),
    },
}

_LABOR_DIVISION_PROPERTY: dict[str, dict[str, str]] = {
    "labor_division_rationale": {
        "type": "string",
        "description": (
            "How you divided labor across agents: which facet each agent owns, why parallel "
            "vs sequential, and why this split is minimal but sufficient."
        ),
    },
}

_CREATION_RATIONALE_PROPERTY: dict[str, dict[str, str]] = {
    "creation_rationale": {
        "type": "string",
        "description": (
            "Why a new agent is needed instead of reusing an existing one, and why this "
            "specialty/base model combination is the right choice."
        ),
    },
}

_RETIREMENT_RATIONALE_PROPERTY: dict[str, dict[str, str]] = {
    "retirement_rationale": {
        "type": "string",
        "description": (
            "Why this agent is being retired (e.g. repeated semantic metric failures, "
            "obsolete specialty, or definitive non-need)."
        ),
    },
}


def ask_agent_rationale_properties() -> dict[str, dict[str, str]]:
    props = dict(_CEO_RATIONALE_BASE_PROPERTIES)
    props.update(_AGENT_SELECTION_PROPERTY)
    return props


def ask_agent_rationale_required() -> list[str]:
    return [
        "ceo_rationale",
        "agent_selection_rationale",
        "semantic_metrics_influence",
    ]


def ask_multiple_agents_rationale_properties() -> dict[str, dict[str, str]]:
    props = dict(_CEO_RATIONALE_BASE_PROPERTIES)
    props.update(_AGENT_SELECTION_PROPERTY)
    props.update(_LABOR_DIVISION_PROPERTY)
    return props


def ask_multiple_agents_rationale_required() -> list[str]:
    return [
        "ceo_rationale",
        "agent_selection_rationale",
        "labor_division_rationale",
        "semantic_metrics_influence",
    ]


def agent_creator_rationale_properties() -> dict[str, dict[str, str]]:
    props = dict(_CEO_RATIONALE_BASE_PROPERTIES)
    props.update(_CREATION_RATIONALE_PROPERTY)
    return props


def agent_creator_rationale_required() -> list[str]:
    return ["ceo_rationale", "creation_rationale", "semantic_metrics_influence"]


def fire_agent_rationale_properties() -> dict[str, dict[str, str]]:
    props = dict(_CEO_RATIONALE_BASE_PROPERTIES)
    props.update(_RETIREMENT_RATIONALE_PROPERTY)
    return props


def fire_agent_rationale_required() -> list[str]:
    return ["ceo_rationale", "retirement_rationale", "semantic_metrics_influence"]


def extract_ceo_decision_fields(args: Any) -> dict[str, Any]:
    """Pull rationale strings from a tool-call args dict for trace rows."""
    if not isinstance(args, dict):
        return {"ceo_rationale_missing": True}
    out: dict[str, Any] = {}
    for key in CEO_RATIONALE_FIELD_NAMES:
        val = args.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            out[key] = text[:_CEO_RATIONALE_STR_LIMIT]
    if not out.get("ceo_rationale"):
        out["ceo_rationale_missing"] = True
    return out


def snapshot_prior_semantic_context() -> dict[str, Any]:
    """
    Summarize the most recent worker round(s) so ceo_decision rows show what
    semantic state the CEO was reacting to.
    """
    st = _get_turn_state()
    history: list = list((st or {}).get("ceo_worker_round_history") or [])
    if not history:
        return {
            "had_prior_worker_round": False,
            "had_recent_semantic_concern": False,
            "prior_round_count": 0,
        }

    last = history[-1] if isinstance(history[-1], dict) else {}
    best = select_best_prior_worker_round(history)
    concerned_rounds = [
        i + 1
        for i, h in enumerate(history)
        if isinstance(h, dict) and h.get("semantic_quality_concern")
    ]
    flagged_agents: list[str] = []
    for h in history:
        if not isinstance(h, dict) or not h.get("semantic_quality_concern"):
            continue
        if h.get("tool") == "AskAgent" and h.get("agent_name"):
            flagged_agents.append(str(h["agent_name"]))
        elif h.get("tool") == "AskMultipleAgents":
            for name in h.get("agents_with_semantic_concern") or []:
                flagged_agents.append(str(name))

    ctx: dict[str, Any] = {
        "had_prior_worker_round": True,
        "prior_round_count": len(history),
        "had_recent_semantic_concern": bool(last.get("semantic_quality_concern")),
        "recent_semantic_concern_rounds": concerned_rounds[-5:],
        "agents_with_recent_semantic_concern": list(dict.fromkeys(flagged_agents))[:12],
        "last_round_tool": last.get("tool"),
        "last_round_agent_name": last.get("agent_name"),
        "last_round_semantic_entropy": last.get("semantic_entropy"),
        "last_round_semantic_density": last.get("semantic_density"),
        "last_round_semantic_quality_concern": last.get("semantic_quality_concern"),
        "last_round_semantic_quality_summary": (last.get("semantic_quality_summary") or "")[
            :2000
        ],
    }
    if isinstance(best, dict):
        ctx["heuristic_best_prior_round_index"] = history.index(best) + 1
        ctx["heuristic_best_prior_entropy"] = best.get("semantic_entropy")
        ctx["heuristic_best_prior_density"] = best.get("semantic_density")
    return ctx


def _append_decision_history(entry: dict[str, Any]) -> None:
    st = _get_turn_state()
    if not st:
        return
    hist = st.setdefault("ceo_decision_history", [])
    if isinstance(hist, list):
        hist.append(entry)


def get_ceo_decision_history() -> list[dict[str, Any]]:
    st = _get_turn_state()
    if not st:
        return []
    hist = st.get("ceo_decision_history")
    return list(hist) if isinstance(hist, list) else []


def log_ceo_decision_before_tool(tool_name: str, args: Any) -> None:
    """Emit ``ceo_decision`` immediately before executing a delegation tool."""
    if tool_name not in CEO_DELEGATION_TOOLS:
        return
    prior = snapshot_prior_semantic_context()
    fields = extract_ceo_decision_fields(args)
    entry = {
        "tool": tool_name,
        "decision_phase": "pre_tool_invoke",
        **fields,
        "prior_semantic_context": prior,
        "responds_to_prior_semantic_concern": prior.get("had_recent_semantic_concern"),
    }
    _append_decision_history(entry)
    log_orchestration_event("ceo_decision", **entry)


def extract_ceo_synthesis_rationale(messages: list[Any]) -> str | None:
    """
    Parse ``CEO_SYNTHESIS_RATIONALE: ...`` from the last non-metadata assistant
    message, if the CEO included it before the final answer.
    """
    marker = "CEO_SYNTHESIS_RATIONALE:"
    for m in reversed(messages or []):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        if m.get("metadata"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if marker in content:
            tail = content.split(marker, 1)[1].strip()
            first_line = tail.split("\n", 1)[0].strip()
            return first_line[:_CEO_RATIONALE_STR_LIMIT] if first_line else None
    return None


def ceo_final_answer_decision_fields(messages: list[Any]) -> dict[str, Any]:
    """Extra trace fields for ``ceo_final_answer`` decision transparency."""
    synthesis = extract_ceo_synthesis_rationale(messages)
    history = get_ceo_decision_history()
    out: dict[str, Any] = {
        "ceo_decision_count_this_turn": len(history),
    }
    if history:
        out["ceo_decision_history"] = history[-12:]
    if synthesis:
        out["ceo_synthesis_rationale"] = synthesis
    else:
        out["ceo_synthesis_rationale_missing"] = True
    return out
