from src.manager.agent_manager import AgentManager
from src.manager.worker_model_policy import policy_summary

__all__ = ['GetAgents']

class GetAgents():
    dependencies = []

    inputSchema = {
        "name": "GetAgents",
        "description": (
            "Retrieves all available AI agents with name, description (specialty), and base_model. "
            "Use this to check whether an existing agent's specialty already covers a task before creating a new agent; "
            "avoid creating near-duplicate agents (e.g. ReservationAssistant vs ReservationInvestigator). "
            "Agents can be invoked with AskAgent (single agent) or AskMultipleAgents (multiple agents for one question)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    def run(self, **kwargs):
        
        agent_manger = AgentManager()
        agents = agent_manger.list_agents()

        return {
            "status": "success",
            "message": "Agents list retrieved successfully",
            "agents": agents,
            "worker_model_policy": policy_summary(),
        }
