from src.manager.agent_manager import AgentManager
from src.manager.orchestration_trace import log_orchestration_event

__all__ = ['FireAgent']

class FireAgent():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "FireAgent",
        "description": "Fires an AI agent for you.",
        "parameters": {
            "type": "object",
            "properties":{
                "agent_name": {
                    "type": "string",
                    "description": "Name of the AI agent that is to be fired. This name cannot have spaces or special characters. It should be a single word.",
                },
            },
            "required": ["agent_name"],
        }
    }

    def run(self, **kwargs):
        print("Running Fire Agent")
        agent_name= kwargs.get("agent_name")

        agent_manager = AgentManager()
                
        try:
            remaining_resource_budget, remaining_expense_budget = agent_manager.delete_agent(agent_name=agent_name)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "output": None
            }

        log_orchestration_event(
            "ceo_fire_agent",
            agent_name=agent_name,
            worker_prompt=None,
            semantic_entropy=None,
            semantic_density=None,
        )
        return {
            "status": "success",
            "message": "Agent successfully fired.",
            "remaining_resource_budget": remaining_resource_budget,
            "remaining_expense_budget": remaining_expense_budget,
        }