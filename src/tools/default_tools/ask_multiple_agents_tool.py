"""
Ask multiple agents for a single user question (e.g. medical + robotics for "robotic doctor").
Per-agent semantic entropy/density are computed per worker row; top-level concern is the OR of
those rows (not the mean of metrics). Optional global scope adds one combined metric.
"""
from src.manager.agent_manager import AgentManager
from src.manager.ceo_decision_trace import (
    ask_multiple_agents_rationale_properties,
    ask_multiple_agents_rationale_required,
    extract_ceo_decision_fields,
)
from src.manager.semantic_ablation import (
    ask_multiple_agents_tool_enabled,
    semantic_threshold_violations,
)
from src.manager.orchestration_trace import (
    ceo_worker_round_cap_tool_result,
    ceo_worker_tool_round_cap_reached,
    log_orchestration_event,
    record_ceo_worker_tool_round,
)

__all__ = ["AskMultipleAgents"]


def _clip_text(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...<truncated {len(s) - n} chars>"


class AskMultipleAgents:
    dependencies = []

    inputSchema = {
        "name": "AskMultipleAgents",
        "description": (
            "Ask multiple AI agents about different aspects of a single question, then get one combined answer and "
            "semantic metrics (entropy/density). semantic_quality_concern is true if any worker row crosses thresholds "
            "(not based on averaged top-level metrics). Default metric_scope is per_agent (safer for different subtasks). "
            "Set metric_scope=global only when agents answer the same question from different viewpoints. Use when the question is "
            "multi-faceted and different existing agents cover different parts (e.g. medical agent + robotics agent for "
            "'robotic doctor'). Pass user_question (the overall question) and agent_prompts_json: a JSON array of "
            "{\"agent_name\": \"...\", \"prompt\": \"...\"} for each agent and their sub-prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                    "description": "The overall user question (used for semantic metrics).",
                },
                "agent_prompts_json": {
                    "type": "string",
                    "description": (
                        "JSON array of objects with agent_name and prompt. "
                        "E.g. [{\"agent_name\": \"MedicalExpert\", \"prompt\": \"What are the clinical aspects?\"}, "
                        "{\"agent_name\": \"RoboticsExpert\", \"prompt\": \"What are the robotics aspects?\"}]"
                    ),
                },
                "metric_scope": {
                    "type": "string",
                    "enum": ["per_agent", "global"],
                    "description": (
                        "Semantic metric scope for AskMultipleAgents. "
                        "per_agent (default): thresholds apply per worker row; top-level entropy/density are means for diagnostics only. "
                        "global: also compute one aggregate metric over the combined multi-agent string."
                    ),
                },
                **ask_multiple_agents_rationale_properties(),
            },
            "required": [
                "agent_prompts_json",
                *ask_multiple_agents_rationale_required(),
            ],
        },
    }

    def run(self, **kwargs):
        import json

        if not ask_multiple_agents_tool_enabled():
            return {
                "status": "error",
                "message": (
                    "AskMultipleAgents is disabled (HASHIRU_ENABLE_ASK_MULTIPLE_AGENTS=0). "
                    "Use AskAgent for each worker."
                ),
                "output": None,
            }

        agent_prompts_json = kwargs.get("agent_prompts_json", "[]")
        user_question = (kwargs.get("user_question") or "").strip()
        metric_scope = (kwargs.get("metric_scope") or "per_agent").strip().lower()

        try:
            agent_prompts = json.loads(agent_prompts_json)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Invalid JSON in agent_prompts_json: {e}",
                "output": None,
            }

        if not isinstance(agent_prompts, list) or len(agent_prompts) == 0:
            return {
                "status": "error",
                "message": "agent_prompts_json must be a non-empty JSON array of {agent_name, prompt}.",
                "output": None,
            }
        if metric_scope not in ("per_agent", "global"):
            return {
                "status": "error",
                "message": "metric_scope must be one of: per_agent, global",
                "output": None,
            }

        if ceo_worker_tool_round_cap_reached():
            return ceo_worker_round_cap_tool_result(requested_tool="AskMultipleAgents")

        agent_manager = AgentManager()
        try:
            result_dict, entropy, density, res_budget, exp_budget = agent_manager.ask_multiple_agents(
                agent_prompts=agent_prompts,
                user_question=user_question,
                metric_scope=metric_scope,
            )
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None,
            }

        print(
            f"[AskMultipleAgents semantic metrics] entropy={result_dict.get('semantic_entropy')!r} "
            f"density={result_dict.get('semantic_density')!r} "
            f"scope={result_dict.get('semantic_metric_scope')!r}"
        )

        per_rows = result_dict.get("per_agent_outputs") or []
        pa_brief = []
        if isinstance(per_rows, list):
            for row in per_rows:
                if not isinstance(row, dict):
                    continue
                pr = str(row.get("prompt") or "")
                rr = str(row.get("response") or "")
                pa_brief.append(
                    {
                        "agent_name": row.get("agent_name"),
                        "prompt": _clip_text(pr, 6000),
                        "response": _clip_text(rr, 12000),
                        "semantic_quality_concern": row.get("semantic_quality_concern"),
                        "semantic_entropy": row.get("semantic_entropy"),
                        "semantic_density": row.get("semantic_density"),
                    }
                )
        any_concern = any(
            isinstance(r, dict) and bool(r.get("semantic_quality_concern")) for r in per_rows
        )
        concern = bool(result_dict.get("semantic_quality_concern", any_concern))
        agents_flagged_list = result_dict.get("agents_with_semantic_concern")
        if not isinstance(agents_flagged_list, list):
            agents_flagged_list = [
                str(r.get("agent_name"))
                for r in per_rows
                if isinstance(r, dict)
                and r.get("agent_name")
                and bool(r.get("semantic_quality_concern"))
            ]

        violations_merged: list[str] = []
        if isinstance(per_rows, list):
            for row in per_rows:
                if not isinstance(row, dict) or not row.get("semantic_quality_concern"):
                    continue
                an = str(row.get("agent_name") or "?")
                for v in semantic_threshold_violations(
                    row.get("semantic_entropy"), row.get("semantic_density")
                ):
                    violations_merged.append(f"{an}:{v}")
        if concern and not violations_merged and metric_scope == "global":
            violations_merged.extend(
                semantic_threshold_violations(
                    result_dict.get("semantic_entropy"),
                    result_dict.get("semantic_density"),
                )
            )

        semantic_quality_summary = ""
        suggested_ceo_next_steps: list[str] = []
        if concern:
            flagged_txt = (
                ", ".join(agents_flagged_list) if agents_flagged_list else "combined output (global)"
            )
            semantic_quality_summary = (
                f"One or more workers crossed semantic thresholds ({flagged_txt}). "
                "Treat per_agent_outputs[].semantic_quality_concern per row; re-prompt failing agents only."
            )
            suggested_ceo_next_steps = [
                (
                    "Re-call AskAgent for each flagged worker: "
                    + ", ".join(agents_flagged_list)
                    + "."
                )
                if agents_flagged_list
                else "Re-run with clearer sub-prompts or adjust metric_scope.",
                "Synthesize after each concerned row reports semantic_quality_concern=false.",
            ]
            if ask_multiple_agents_tool_enabled():
                suggested_ceo_next_steps.append(
                    "Optionally use AskMultipleAgents again with tighter instructions for the failing role(s) only."
                )
            else:
                suggested_ceo_next_steps.append(
                    "Use AskAgent per failing role with tighter sub-prompts (AskMultipleAgents is disabled)."
                )

        co = str(result_dict.get("combined_output") or "")
        uq = (user_question or "").strip() or (
            (agent_prompts[0].get("prompt", "")) if agent_prompts else ""
        )
        log_orchestration_event(
            "ceo_ask_multiple_agents",
            worker_routing="AskMultipleAgents",
            phase="after_worker_completion",
            user_question=uq,
            metric_scope=result_dict.get("semantic_metric_scope", metric_scope),
            semantic_entropy=result_dict.get("semantic_entropy"),
            semantic_density=result_dict.get("semantic_density"),
            semantic_quality_concern=concern,
            agents_with_semantic_concern=agents_flagged_list,
            **extract_ceo_decision_fields(kwargs),
        )
        record_ceo_worker_tool_round(
            {
                "tool": "AskMultipleAgents",
                "user_question": _clip_text(uq, 8000),
                "combined_output": _clip_text(co, 24000),
                "per_agent_outputs_brief": pa_brief,
                "semantic_quality_concern": concern,
                "semantic_entropy": result_dict.get("semantic_entropy"),
                "semantic_density": result_dict.get("semantic_density"),
                "semantic_metric_scope": result_dict.get("semantic_metric_scope", metric_scope),
                "semantic_quality_summary": semantic_quality_summary,
                "primary_output_excerpt": _clip_text(co, 12000),
                "agents_with_semantic_concern": agents_flagged_list,
            }
        )

        msg = (
            "Multiple agents replied, but one or more outputs crossed semantic thresholds; "
            "review semantic_quality_* and per_agent_outputs—re-prompt concerned agents only."
            if concern
            else (
                "Multiple agents replied; semantic metrics computed with "
                f"metric_scope={result_dict.get('semantic_metric_scope', metric_scope)}."
            )
        )
        return {
            "status": "success",
            "message": msg,
            "output": result_dict.get("combined_output"),
            "combined_output": result_dict.get("combined_output"),
            "per_agent_outputs": result_dict.get("per_agent_outputs", []),
            "semantic_entropy": result_dict.get("semantic_entropy"),
            "semantic_density": result_dict.get("semantic_density"),
            "semantic_metric_scope": result_dict.get("semantic_metric_scope", metric_scope),
            "semantic_quality_concern": concern,
            "agents_with_semantic_concern": agents_flagged_list,
            "semantic_threshold_violations": violations_merged,
            "semantic_quality_summary": semantic_quality_summary,
            "suggested_ceo_next_steps": suggested_ceo_next_steps,
            "remaining_resource_budget": res_budget,
            "remaining_expense_budget": exp_budget,
        }
