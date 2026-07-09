from src.manager.agent_manager import AgentManager
from src.manager.semantic_ablation import ask_multiple_agents_tool_enabled
from src.manager.worker_model_policy import policy_summary

__all__ = ["GetAgents"]


def _get_agents_input_schema() -> dict:
    base = (
        "Retrieves all available AI agents with name, description (specialty), and base_model. "
        "Use this to check whether an existing agent's specialty already covers a task before creating a new agent; "
        "avoid creating near-duplicate agents (e.g. ReservationAssistant vs ReservationInvestigator). "
    )
    if ask_multiple_agents_tool_enabled():
        tail = (
            "Agents can be invoked with AskAgent (single agent) or AskMultipleAgents "
            "(multiple agents for one question)."
        )
    else:
        tail = (
            "Invoke workers with AskAgent (one agent per call). AskMultipleAgents is disabled "
            "(HASHIRU_ENABLE_ASK_MULTIPLE_AGENTS=0)."
        )
    return {
        "name": "GetAgents",
        "description": base + tail,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class GetAgents:
    dependencies = []

    inputSchema = _get_agents_input_schema()

    def run(self, **kwargs):

        agent_manager = AgentManager()
        agents = agent_manager.list_agents()

        return {
            "status": "success",
            "message": "Agents list retrieved successfully",
            "agents": agents,
            "worker_model_policy": policy_summary(),
        }
