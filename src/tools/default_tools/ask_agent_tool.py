from src.manager.agent_manager import AgentManager, compact_worker_reply_for_ceo
from src.manager.ceo_decision_trace import (
    ask_agent_rationale_properties,
    ask_agent_rationale_required,
    extract_ceo_decision_fields,
)
from src.manager.orchestration_trace import (
    ceo_worker_round_cap_tool_result,
    ceo_worker_tool_round_cap_reached,
    log_orchestration_event,
    record_ceo_worker_tool_round,
)

__all__ = ['AskAgent']


def _clip_text(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...<truncated {len(s) - n} chars>"


class AskAgent():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "AskAgent",
        "description": (
            "Asks an AI agent a question and gets a response. The agent must be created using the AgentCreator tool first. "
            "If semantic_quality_concern is true in the result, the first answer crossed entropy/density thresholds—YOU (CEO) must "
            "decide the next step (reprompt, AskMultipleAgents, different agent, retire worker, etc.); the worker is not auto re-prompted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the AI agent that is to be asked a question. This name cannot have spaces or special characters. It should be a single word.",
                },
                "prompt": {
                    "type": "string",
                    "description": "This is the prompt that will be used to ask the agent a question. It should be a string that describes the question to be asked.",
                },
                **ask_agent_rationale_properties(),
            },
            "required": ["agent_name", "prompt", *ask_agent_rationale_required()],
        }
    }

    def run(self, **kwargs):
        print("Asking agent a question")

        agent_name = kwargs.get("agent_name")
        prompt = kwargs.get("prompt")
        if ceo_worker_tool_round_cap_reached():
            return ceo_worker_round_cap_tool_result(requested_tool="AskAgent")

        agent_manger = AgentManager()

        try:
            agent_response, remaining_resource_budget, remaining_expense_budget, orch_meta = agent_manger.ask_agent(
                agent_name=agent_name, prompt=prompt
            )
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "output": None
            }

        print("Agent response", compact_worker_reply_for_ceo(agent_response))
        print(
            f"[AskAgent semantic metrics] entropy={orch_meta.get('semantic_entropy')!r} "
            f"density={orch_meta.get('semantic_density')!r} "
            f"extra_samples_used={orch_meta.get('num_stochastic_samples_first_round')!r} "
            f"entropy_ok={(orch_meta.get('metrics_diagnostics_final') or {}).get('entropy_ok')!r} "
            f"density_ok={(orch_meta.get('metrics_diagnostics_final') or {}).get('density_ok')!r}"
        )
        concern = bool(orch_meta.get("semantic_quality_concern"))
        _ceo_ask_extras = {}
        _sar = orch_meta.get("semantic_auxiliary_responses")
        if isinstance(_sar, list):
            _ceo_ask_extras["semantic_auxiliary_responses"] = _sar
        log_orchestration_event(
            "ceo_ask_agent",
            worker_routing="AskAgent",
            phase="after_worker_completion",
            agent_name=agent_name,
            worker_prompt=prompt,
            **extract_ceo_decision_fields(kwargs),
            worker_response=orch_meta.get("worker_response"),
            worker_response_kind=orch_meta.get("worker_response_kind"),
            semantic_auxiliary_completions_count=orch_meta.get(
                "semantic_auxiliary_completions_count"
            ),
            semantic_metrics_include_auxiliary_samples=orch_meta.get(
                "semantic_metrics_include_auxiliary_samples"
            ),
            same_prompt_completion_index_sent_to_ceo=orch_meta.get(
                "same_prompt_completion_index_sent_to_ceo"
            ),
            same_prompt_total_llm_completions=orch_meta.get(
                "same_prompt_total_llm_completions"
            ),
            semantic_entropy=orch_meta.get("semantic_entropy"),
            semantic_density=orch_meta.get("semantic_density"),
            worker_invocation_index=orch_meta.get("worker_invocation_index"),
            worker_reprompted_after_semantic_check=orch_meta.get(
                "worker_reprompted_after_semantic_check"
            ),
            semantic_quality_concern=concern,
            base_model=orch_meta.get("base_model"),
            semantic_ablation=orch_meta.get("semantic_ablation"),
            **_ceo_ask_extras,
        )
        msg = (
            "Agent replied, but semantic entropy/density crossed thresholds; review semantic_quality_* fields and choose the next action."
            if concern
            else "Agent has replied to the given prompt"
        )
        wr = str(orch_meta.get("worker_response") or "")
        record_ceo_worker_tool_round(
            {
                "tool": "AskAgent",
                "agent_name": agent_name,
                "worker_prompt": _clip_text(str(prompt or ""), 12000),
                "worker_response": _clip_text(wr, 24000),
                "semantic_quality_concern": concern,
                "semantic_entropy": orch_meta.get("semantic_entropy"),
                "semantic_density": orch_meta.get("semantic_density"),
                "semantic_quality_summary": orch_meta.get("semantic_quality_summary", ""),
                "primary_output_excerpt": _clip_text(wr, 12000),
            }
        )
        return {
            "status": "success",
            "message": msg,
            "output": compact_worker_reply_for_ceo(agent_response),
            "remaining_resource_budget": remaining_resource_budget,
            "remaining_expense_budget": remaining_expense_budget,
            "semantic_quality_concern": concern,
            "semantic_threshold_violations": orch_meta.get("semantic_threshold_violations", []),
            "semantic_quality_summary": orch_meta.get("semantic_quality_summary", ""),
            "suggested_ceo_next_steps": orch_meta.get("suggested_ceo_next_steps", []),
            "worker_prompt_excerpt_for_ceo": orch_meta.get("worker_prompt_excerpt_for_ceo", ""),
            "orchestration_meta": orch_meta,
        }
