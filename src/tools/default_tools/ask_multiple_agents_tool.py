"""
Ask multiple agents for a single user question (e.g. medical + robotics for "robotic doctor").
Semantic entropy and density are computed over ALL responses from ALL agents.
"""
from src.manager.agent_manager import AgentManager

__all__ = ["AskMultipleAgents"]


class AskMultipleAgents:
    dependencies = []

    inputSchema = {
        "name": "AskMultipleAgents",
        "description": (
            "Ask multiple AI agents about different aspects of a single question, then get one combined answer and "
            "semantic metrics (entropy/density) computed over ALL agents' responses. Use when the question is "
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
            },
            "required": ["agent_prompts_json"],
        },
    }

    def run(self, **kwargs):
        import json
        agent_prompts_json = kwargs.get("agent_prompts_json", "[]")
        user_question = (kwargs.get("user_question") or "").strip()

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

        agent_manager = AgentManager()
        try:
            result_dict, entropy, density, res_budget, exp_budget = agent_manager.ask_multiple_agents(
                agent_prompts=agent_prompts,
                user_question=user_question,
            )
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None,
            }

        return {
            "status": "success",
            "message": "Multiple agents replied; semantic metrics are over all agents' responses.",
            "output": result_dict.get("combined_output"),
            "combined_output": result_dict.get("combined_output"),
            "per_agent_outputs": result_dict.get("per_agent_outputs", []),
            "semantic_entropy": result_dict.get("semantic_entropy"),
            "semantic_density": result_dict.get("semantic_density"),
            "remaining_resource_budget": res_budget,
            "remaining_expense_budget": exp_budget,
        }
