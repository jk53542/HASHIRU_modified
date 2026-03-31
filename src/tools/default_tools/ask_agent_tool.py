from src.manager.agent_manager import AgentManager, compact_worker_reply_for_ceo
from src.manager.orchestration_trace import log_orchestration_event

__all__ = ['AskAgent']


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
                }
            },
            "required": ["agent_name", "prompt"],
        }
    }

    def run(self, **kwargs):
        print("Asking agent a question")

        agent_name = kwargs.get("agent_name")
        prompt = kwargs.get("prompt")
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
        log_orchestration_event(
            "ceo_ask_agent",
            agent_name=agent_name,
            worker_prompt=prompt,
            semantic_entropy=orch_meta.get("semantic_entropy"),
            semantic_density=orch_meta.get("semantic_density"),
            worker_invocation_index=orch_meta.get("worker_invocation_index"),
            worker_reprompted_after_semantic_check=orch_meta.get(
                "worker_reprompted_after_semantic_check"
            ),
            semantic_quality_concern=concern,
            base_model=orch_meta.get("base_model"),
            semantic_ablation=orch_meta.get("semantic_ablation"),
        )
        msg = (
            "Agent replied, but semantic entropy/density crossed thresholds; review semantic_quality_* fields and choose the next action."
            if concern
            else "Agent has replied to the given prompt"
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
