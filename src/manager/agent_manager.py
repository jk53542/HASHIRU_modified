from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, Tuple
import os
import json
import ollama
from openai import OpenAI
from src.manager.utils.singleton import singleton
from src.manager.utils.streamlit_interface import output_assistant_response
from google import genai
from google.genai import types
from google.genai.types import *
from groq import Groq
import os
from dotenv import load_dotenv
from src.manager.budget_manager import BudgetManager
import logging
import time

# @@ inside src/manager/agent_manager.py
from src.manager.tool_manager import ToolManager  # if ToolManager is importable here
# OR direct import of metrics wrapper:
# from src.metrics.semantic_metrics import compute_semantic_entropy, compute_semantic_density

MODEL_PATH = "./src/models/"
MODEL_FILE_PATH = "./src/models/models.json"

SEMANTIC_ENTROPY_THRESHOLD = 1.0  # high value -> uncertain, will need to tune this!!
SEMANTIC_DENSITY_THRESHOLD = 0.3  # low value -> low confidence, will need to tune this!!


class Agent(ABC):

    def __init__(self, agent_name: str,
                 base_model: str,
                 system_prompt: str,
                 create_resource_cost: int,
                 invoke_resource_cost: int,
                 create_expense_cost: int = 0,
                 invoke_expense_cost: int = 0,
                 output_expense_cost: int = 0):
        self.agent_name = agent_name
        self.base_model = base_model
        self.system_prompt = system_prompt
        self.create_resource_cost = create_resource_cost
        self.invoke_resource_cost = invoke_resource_cost
        self.create_expense_cost = create_expense_cost
        self.invoke_expense_cost = invoke_expense_cost
        self.output_expense_cost = output_expense_cost
        self.create_model()

    @abstractmethod
    def create_model(self) -> None:
        """Create and Initialize agent"""
        pass

    @abstractmethod
    def ask_agent(self, prompt: str) -> str:
        """ask agent a question"""
        pass

    @abstractmethod
    def delete_agent(self) -> None:
        """delete agent"""
        pass

    @abstractmethod
    def get_type(self) -> None:
        """get agent type"""
        pass

    def get_costs(self):
        return {
            "create_resource_cost": self.create_resource_cost,
            "invoke_resource_cost": self.invoke_resource_cost,
            "create_expense_cost": self.create_expense_cost,
            "invoke_expense_cost": self.invoke_expense_cost,
            "output_expense_cost": self.output_expense_cost,
        }


    # @@ NEW ADDITIONS, keep an eye out on these next few functions in case they break things
    def append_to_history(self, entry: dict):
        """
        Append a single history entry (expected form used earlier in integration):
        {
            "role": "agent" or "user",
            "text": "...",
            "semantic_entropy": float or None,
            "semantic_density": float or None,
            "timestamp": float
        }
        This method is intentionally tolerant (does minimal validation).
        """
        try:
            if not isinstance(entry, dict):
                # try to coerce simple strings into a text entry
                entry = {"role": "agent", "text": str(entry), "timestamp": time.time()}
            # Ensure there is a timestamp
            if "timestamp" not in entry:
                entry["timestamp"] = time.time()

            # Append to in-memory history
            self.history.append(entry)

            # Optionally: persist to global MemoryManager if available
            try:
                # avoid import cycles; import lazily
                from src.manager.memory_manager import MemoryManager
                mm = MemoryManager.get_instance()
                # store only high-level entries to memory (you can tune keys)
                # Example criteria: store only agent outputs with semantic metrics or user preference entries
                if entry.get("role") == "agent":
                    # store compact memory record to avoid large memory bloat
                    mem = {
                        "type": "agent_output",
                        "agent": getattr(self, "name", self.__class__.__name__),
                        "text": entry.get("text"),
                        "semantic_entropy": entry.get("semantic_entropy"),
                        "semantic_density": entry.get("semantic_density"),
                        "timestamp": entry.get("timestamp")
                    }
                    # MemoryManager may implement add_memory(action/add_memory) interface — be defensive
                    try:
                        mm.add_memory(mem)
                    except Exception:
                        # if memory manager API differs, ignore failure
                        pass
            except Exception:
                # If MemoryManager not available (or import cycle), ignore gracefully
                pass

        except Exception as e:
            # append_to_history should never raise; just log
            try:
                self.logger.exception("append_to_history failed: %s", e)
            except Exception:
                # worst case: fallback to module logger
                _logger.exception("append_to_history failed: %s", e)

    def get_history(self, n: int | None = None):
        if n is None:
            return list(self.history)
        if n <= 0:
            return []
        return self.history[-n:]

    def clear_history(self):
        self.history = []


class OllamaAgent(Agent):
    type = "local"

    def create_model(self):
        ollama_response = ollama.create(
            model=self.agent_name,
            from_=self.base_model,
            system=self.system_prompt,
            stream=False
        )

    def ask_agent(self, prompt):
        output_assistant_response(f"Asked Agent {self.agent_name} a question")
        agent_response = ollama.chat(
            model=self.agent_name,
            messages=[{"role": "user", "content": prompt}],
        )
        output_assistant_response(
            f"Agent {self.agent_name} answered with {agent_response.message.content}")
        return agent_response.message.content

    def delete_agent(self):
        ollama.delete(self.agent_name)

    def get_type(self):
        return self.type


class GeminiAgent(Agent):
    type = "cloud"

    def __init__(self,
                 agent_name: str,
                 base_model: str,
                 system_prompt: str,
                 create_resource_cost: int,
                 invoke_resource_cost: int,
                 create_expense_cost: int = 0,
                 invoke_expense_cost: int = 0,
                 output_expense_cost: int = 0):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required for Gemini models. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

        # Initialize the Gemini API
        self.client = genai.Client(api_key=self.api_key)
        self.chat = self.client.chats.create(model=base_model)

        # Call parent constructor after API setup
        super().__init__(agent_name,
                         base_model,
                         system_prompt,
                         create_resource_cost,
                         invoke_resource_cost,
                         create_expense_cost,
                         invoke_expense_cost,
                         output_expense_cost)

    def create_model(self):
        self.messages = []

    def ask_agent(self, prompt):
        response = self.chat.send_message(
            message=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
            )
        )
        return response.text

    def delete_agent(self):
        self.messages = []

    def get_type(self):
        return self.type


class GroqAgent(Agent):
    type = "cloud"

    def __init__(
        self,
        agent_name: str,
        base_model: str,
        system_prompt: str,
        create_resource_cost: int,
        invoke_resource_cost: int,
        create_expense_cost: int = 0,
        invoke_expense_cost: int = 0,
        output_expense_cost: int = 0
    ):
        # Call the parent class constructor first
        super().__init__(agent_name, base_model, system_prompt,
                         create_resource_cost, invoke_resource_cost,
                         create_expense_cost, invoke_expense_cost,
                         output_expense_cost)

        # Groq-specific API client setup
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file or environment.")
        self.client = Groq(api_key=api_key)

        if self.base_model and "groq-" in self.base_model:
            self.groq_api_model_name = self.base_model.split("groq-", 1)[1]
        else:
            # Fallback or error if the naming convention isn't followed.
            # This ensures that if a non-prefixed model name is somehow passed,
            # it might still work, or you can raise an error.
            self.groq_api_model_name = self.base_model
            print(f"Warning: GroqAgent base_model '{self.base_model}' does not follow 'groq-' prefix convention.")

    def create_model(self) -> None:
        """
        Create and Initialize agent.
        For Groq, models are pre-existing on their cloud.
        This method is called by Agent's __init__.
        """
        pass

    def ask_agent(self, prompt: str) -> str:
        """Ask agent a question"""
        if not self.client:
            raise ConnectionError("Groq client not initialized. Check API key and constructor.")
        if not self.groq_api_model_name:
            raise ValueError("Groq API model name not set. Check base_model configuration.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.groq_api_model_name, # Use the derived model name for Groq API
            )
            result = response.choices[0].message.content
            return result
        except Exception as e:
            # Handle API errors or other exceptions during the call
            print(f"Error calling Groq API: {e}")
            raise  # Re-raise the exception or handle it as appropriate

    def delete_agent(self) -> None:
        """Delete agent"""
        pass

    def get_type(self) -> str: # Ensure return type hint matches Agent ABC
        """Get agent type"""
        return self.type

class LambdaAgent(Agent):
    type = "cloud"

    def __init__(self,
                 agent_name: str,
                 base_model: str,
                 system_prompt: str,
                 create_resource_cost: int,
                 invoke_resource_cost: int,
                 create_expense_cost: int = 0,
                 invoke_expense_cost: int = 0,
                 output_expense_cost: int = 0,
                 api_key: str = ""):

        self.lambda_url = "https://api.lambda.ai/v1"
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")

        self.lambda_model = base_model.split("lambda-")[1] if base_model.startswith("lambda-") else base_model
        if not self.api_key:
            raise ValueError("Lambda API key must be provided or set in LAMBDA_API_KEY environment variable.")
        
        self.client = client = OpenAI(
            api_key=self.api_key,
            base_url=self.lambda_url,
        )

        super().__init__(agent_name,
                         base_model,
                         system_prompt,
                         create_resource_cost,
                         invoke_resource_cost,
                         create_expense_cost,
                         invoke_expense_cost,
                         output_expense_cost)

    def create_model(self) -> None:
        pass  # Lambda already deployed

    def ask_agent(self, prompt: str) -> str:
        """Ask agent a question"""
        try:
            response = self.client.chat.completions.create(
                model=self.lambda_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            output_assistant_response(f"Error asking agent: {e}")
            raise

    def delete_agent(self) -> None:
        pass

    def get_type(self) -> str:
        return self.type

@singleton
class AgentManager():
    budget_manager: BudgetManager = BudgetManager()
    is_creation_enabled: bool = True
    is_cloud_invocation_enabled: bool = True
    is_local_invocation_enabled: bool = True

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self.logger = logging.getLogger("AgentManager")
        self._agent_types = {
            "ollama": OllamaAgent,
            # @@ something weird is going on with the google api key, disable gemini agents for now
            # "gemini": GeminiAgent,
            "groq": GroqAgent,
            "lambda": LambdaAgent,
        }

        self._load_agents()

    def set_creation_mode(self, status: bool):
        self.is_creation_enabled = status
        if status:
            output_assistant_response("Agent creation mode is enabled.")
        else:
            output_assistant_response("Agent creation mode is disabled.")

    def set_cloud_invocation_mode(self, status: bool):
        self.is_cloud_invocation_enabled = status
        if status:
            output_assistant_response("Cloud invocation mode is enabled.")
        else:
            output_assistant_response("Cloud invocation mode is disabled.")

    def set_local_invocation_mode(self, status: bool):
        self.is_local_invocation_enabled = status
        if status:
            output_assistant_response("Local invocation mode is enabled.")
        else:
            output_assistant_response("Local invocation mode is disabled.")

    def create_agent(self, agent_name: str,
                     base_model: str, system_prompt: str,
                     description: str = "", create_resource_cost: float = 0,
                     invoke_resource_cost: float = 0,
                     create_expense_cost: float = 0,
                     invoke_expense_cost: float = 0,
                     output_expense_cost: float = 0,
                     **additional_params) -> Tuple[Agent, int]:
        if not self.is_creation_enabled:
            raise ValueError("Agent creation mode is disabled.")

        if agent_name in self._agents:
            raise ValueError(f"Agent {agent_name} already exists")

        # @@ NEW ADDITION, should throw an error if something is wrong
        if "gemini" in base_model and not os.environ.get("GOOGLE_API_KEY"):
            return {
                "status": "error",
                "message": "Gemini models disabled (GOOGLE_API_KEY not set).",
                "output": None
            }

        self._agents[agent_name] = self.create_agent_class(
            agent_name,
            base_model,
            system_prompt,
            description=description,
            create_resource_cost=create_resource_cost,
            invoke_resource_cost=invoke_resource_cost,
            create_expense_cost=create_expense_cost,
            invoke_expense_cost=invoke_expense_cost,
            output_expense_cost=output_expense_cost,
            **additional_params  # For any future parameters we might want to add
        )

        # save agent to file
        self._save_agent(
            agent_name,
            base_model,
            system_prompt,
            description=description,
            create_resource_cost=create_resource_cost,
            invoke_resource_cost=invoke_resource_cost,
            create_expense_cost=create_expense_cost,
            invoke_expense_cost=invoke_expense_cost,
            output_expense_cost=output_expense_cost,
            **additional_params  # For any future parameters we might want to add
        )
        return (self._agents[agent_name],
                self.budget_manager.get_current_remaining_resource_budget(),
                self.budget_manager.get_current_remaining_expense_budget())

    def validate_budget(self,
                        resource_cost: float = 0,
                        expense_cost: float = 0) -> None:
        if not self.budget_manager.can_spend_resource(resource_cost):
            raise ValueError(f"Do not have enough resource budget to create/use the agent. "
                             + f"Creating/Using the agent costs {resource_cost} but only {self.budget_manager.get_current_remaining_resource_budget()} is remaining")
        if not self.budget_manager.can_spend_expense(expense_cost):
            raise ValueError(f"Do not have enough expense budget to create/use the agent. "
                             + f"Creating/Using the agent costs {expense_cost} but only {self.budget_manager.get_current_remaining_expense_budget()} is remaining")

    def create_agent_class(self,
                           agent_name: str,
                           base_model: str,
                           system_prompt: str,
                           description: str = "",
                           create_resource_cost: float = 0,
                           invoke_resource_cost: float = 0,
                           create_expense_cost: float = 0,
                           invoke_expense_cost: float = 0,
                           output_expense_cost: float = 0,
                           **additional_params) -> Agent:
        agent_type = self._get_agent_type(base_model)
        agent_class = self._agent_types.get(agent_type)

        if not agent_class:
            raise ValueError(f"Unsupported base model {base_model}")

        created_agent = agent_class(agent_name,
                                    base_model,
                                    system_prompt,
                                    create_resource_cost,
                                    invoke_resource_cost,
                                    create_expense_cost,
                                    invoke_expense_cost,
                                    output_expense_cost,
                                    **additional_params)

        self.validate_budget(create_resource_cost,
                             create_expense_cost)

        self.budget_manager.add_to_resource_budget(create_resource_cost)
        self.budget_manager.add_to_expense_budget(create_expense_cost)
        # create agent
        return created_agent

    def get_agent(self, agent_name: str) -> Agent:
        """Get existing agent by name"""
        if agent_name not in self._agents:
            raise ValueError(f"Agent {agent_name} does not exists")
        return self._agents[agent_name]

    def list_agents(self) -> dict:
        """Return agent information (name, description, costs)"""
        try:
            if os.path.exists(MODEL_FILE_PATH):
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    full_models = json.loads(f.read())

                # Create a simplified version with only the description and costs
                simplified_agents = {}
                for name, data in full_models.items():
                    simplified_agents[name] = {
                        "description": data.get("description", ""),
                        "create_resource_cost": data.get("create_resource_cost", 0),
                        "invoke_resource_cost": data.get("invoke_resource_cost", 0),
                        "create_expense_cost": data.get("create_expense_cost", 0),
                        "invoke_expense_cost": data.get("invoke_expense_cost", 0),
                        "base_model": data.get("base_model", ""),
                    }
                return simplified_agents
            else:
                return {}
        except Exception as e:
            output_assistant_response(f"Error listing agents: {e}")
            return {}

    def delete_agent(self, agent_name: str) -> int:
        agent: Agent = self.get_agent(agent_name)

        self.budget_manager.remove_from_resource_expense(
            agent.create_resource_cost)
        agent.delete_agent()

        del self._agents[agent_name]
        try:
            if os.path.exists(MODEL_FILE_PATH):
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    models = json.loads(f.read())

                del models[agent_name]
                with open(MODEL_FILE_PATH, "w", encoding="utf8") as f:
                    f.write(json.dumps(models, indent=4))
        except Exception as e:
            output_assistant_response(f"Error deleting agent: {e}")
        return (self.budget_manager.get_current_remaining_resource_budget(),
                self.budget_manager.get_current_remaining_expense_budget())

    def ask_agent(self, agent_name: str, prompt: str) -> Tuple[str, int]:
        agent: Agent = self.get_agent(agent_name)
        print(agent.get_type())
        print(agent_name)
        print(self.is_local_invocation_enabled,
              self.is_cloud_invocation_enabled)
        if not self.is_local_invocation_enabled and agent.get_type() == "local":
            raise ValueError("Local invocation mode is disabled.")

        if not self.is_cloud_invocation_enabled and agent.get_type() == "cloud":
            raise ValueError("Cloud invocation mode is disabled.")

        n_tokens = len(prompt.split())/1000000

        self.validate_budget(agent.invoke_resource_cost,
                             agent.invoke_expense_cost*n_tokens)

        self.budget_manager.add_to_expense_budget(
            agent.invoke_expense_cost*n_tokens)

        result = agent.ask_agent(prompt)
        n_tokens = len(result.split())/1000000
        self.budget_manager.add_to_expense_budget(
            agent.output_expense_cost*n_tokens)

        # @@ NEW STUFF HERE
        text = result.get("text") if isinstance(result, dict) else result

        # --- Compute semantic metrics (synchronous) ---
        # Prefer calling ToolManager.runTool so we keep the tool semantics (or call direct wrappers).
        try:
            # Option A: call the tool pipeline (preferred if you want CEO to also see a tool call trace)
            # tool_resp = ToolManager.get_instance().runTool("compute_semantic_metrics",
            #                                                {"prompt": prompt, "response": text, "mode":"fast"})
            # entropy = float(tool_resp["entropy"]); density = float(tool_resp["density"])

            # Option B: call directly (faster, bypasses tool code)
            entropy_info = compute_semantic_entropy(prompt=prompt, response=text)
            density_info = compute_semantic_density(prompt=prompt, response=text)
            entropy = entropy_info["entropy"]
            density = density_info["density"]
        except Exception as e:
            # fallback: log and continue without metrics
            self.logger.warning("Semantic metric compute failed for agent %s: %s", agent_name, e)
            entropy = None
            density = None

        # store metrics in agent history for later aggregate scoring
        # @@ source of error, function does not exist for all type of agents, FIX THAT
        # agent.append_to_history({
        #     "role": "agent",
        #     "text": text,
        #     "semantic_entropy": entropy,
        #     "semantic_density": density,
        #     "timestamp": time.time()
        # })

        # @@ NEW ATTEMPT, should log issues instead of crash if the history function does not work
        # After receiving text from agent
        entry = {
            "role": "agent",
            "text": text,
            "semantic_entropy": entropy,
            "semantic_density": density,
            "timestamp": time.time()
        }
        # Defensive call; prefer agent.append_to_history if available
        try:
            append_fn = getattr(agent, "append_to_history", None)
            if callable(append_fn):
                append_fn(entry)
            else:
                # fallback: record in AgentManager-level history map if needed
                try:
                    self._agent_histories.setdefault(agent_id, []).append(entry)
                except Exception:
                    # As last resort, log and continue
                    self.logger.warning("Could not append agent history for %s", getattr(agent, "name", agent_id))
        except Exception as e:
            self.logger.exception("append_to_history invocation failed: %s", e)

        # --- Decision rules: reprompt / escalate ---
        # Example rule: if entropy is high OR density is low, attempt a reprompt
        reprompted = False
        if entropy is not None and density is not None:
            if (entropy > SEMANTIC_ENTROPY_THRESHOLD) or (density < SEMANTIC_DENSITY_THRESHOLD):
                # Prepare a reprompt template. You can make this more sophisticated.
                reprompt_msg = ("Your previous response seems uncertain/conflicting (semantic_entropy={:.3f}, semantic_density={:.3f}). "
                                "Please try again, prioritize factual grounding and be explicit about uncertainty. "
                                "If you can't be confident, say 'I don't know'.\n\nOriginal task: {}\n").format(entropy, density, prompt)
                # Try one reprompt (avoid infinite loop): may pass `reprompt_count` in kwargs to limit
                reprompt_count = kwargs.get("reprompt_count", 0)
                if reprompt_count < 1:
                    reprompted = True
                    new_result = agent.ask_agent(prompt + "\n\n" + reprompt_msg, reprompt_count=reprompt_count+1)
                    # re-evaluate metrics for the new result (optionally)
                    # ... compute metrics again and store
                    result = new_result

        return (result,
                self.budget_manager.get_current_remaining_resource_budget(),
                self.budget_manager.get_current_remaining_expense_budget())

    # @@ NEW FUNCTION
    def evaluate_agent_score(self, agent_name, recent_n=10):
        """
        Existing evaluation already considers cost, performance, and resource usage.
        Add semantic-confidence score into final weighted score.
        """
        base_score = self.compute_base_score(agent_name)
        # compute average density / entropy over recent outputs
        hist = self.get_agent_history(agent_name)[-recent_n:]
        densities = [h.get("semantic_density") for h in hist if h.get("semantic_density") is not None]
        entropies = [h.get("semantic_entropy") for h in hist if h.get("semantic_entropy") is not None]
        if densities:
            avg_density = sum(densities)/len(densities)
        else:
            avg_density = None
        if entropies:
            avg_entropy = sum(entropies)/len(entropies)
        else:
            avg_entropy = None

        # Example weighting: increase score if high density, decrease if high entropy
        score = base_score
        if avg_density is not None:
            score += (avg_density - 0.5) * 2.0   # @@ adjust weight as needed, will need to finetune!!
        if avg_entropy is not None:
            score -= (avg_entropy - 0.5) * 1.5  # @@ penalize high entropy, will need to finetune!!

        # Combine with cost penalties (already in base_score)
        return score

    def _save_agent(self,
                    agent_name: str,
                    base_model: str,
                    system_prompt: str,
                    description: str = "",
                    create_resource_cost: float = 0,
                    invoke_resource_cost: float = 0,
                    create_expense_cost: float = 0,
                    invoke_expense_cost: float = 0,
                    output_expense_cost: float = 0,
                    **additional_params) -> None:
        """Save a single agent to the models.json file"""
        try:
            # Ensure the directory exists
            os.makedirs(MODEL_PATH, exist_ok=True)

            # Read existing models file or create empty dict if it doesn't exist
            try:
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    models = json.loads(f.read())
            except (FileNotFoundError, json.JSONDecodeError):
                models = {}

            # Update the models dict with the new agent
            models[agent_name] = {
                "base_model": base_model,
                "description": description,
                "system_prompt": system_prompt,
                "create_resource_cost": create_resource_cost,
                "invoke_resource_cost": invoke_resource_cost,
                "create_expense_cost": create_expense_cost,
                "invoke_expense_cost": invoke_expense_cost,
                "output_expense_cost": output_expense_cost,
            }

            # Add any additional parameters that were passed
            for key, value in additional_params.items():
                models[agent_name][key] = value

            # Write the updated models back to the file
            with open(MODEL_FILE_PATH, "w", encoding="utf8") as f:
                f.write(json.dumps(models, indent=4))

        except Exception as e:
            output_assistant_response(f"Error saving agent {agent_name}: {e}")

    def _get_agent_type(self, base_model) -> str:
        if base_model == "llama3.2":
            return "ollama"
        elif base_model == "mistral":
            return "ollama"
        elif base_model == "deepseek-r1":
            return "ollama"
        elif "gemini" in base_model:
            return "gemini"
        elif "groq" in base_model:
            return "groq"
        elif base_model.startswith("lambda-"):
            return "lambda"
        else:
            return "unknown"


    def _load_agents(self) -> None:
        """Load agent configurations from disk"""
        try:
            if not os.path.exists(MODEL_FILE_PATH):
                return

            with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                models = json.loads(f.read())

            for name, data in models.items():
                if name in self._agents:
                    continue
                base_model = data["base_model"]
                system_prompt = data["system_prompt"]
                create_resource_cost = data.get("create_resource_cost", 0)
                invoke_resource_cost = data.get("invoke_resource_cost", 0)
                create_expense_cost = data.get("create_expense_cost", 0)
                invoke_expense_cost = data.get("invoke_expense_cost", 0)
                output_expense_cost = data.get("output_expense_cost", 0)
                model_type = self._get_agent_type(base_model)
                manager_class = self._agent_types.get(model_type)

                if manager_class:
                    # Create the agent with the appropriate manager class
                    self._agents[name] = self.create_agent_class(
                        name,
                        base_model,
                        system_prompt,
                        description=data.get("description", ""),
                        create_resource_cost=create_resource_cost,
                        invoke_resource_cost=invoke_resource_cost,
                        create_expense_cost=create_expense_cost,
                        invoke_expense_cost=invoke_expense_cost,
                        output_expense_cost=output_expense_cost,
                        **data.get("additional_params", {})
                    )
                    self._agents[name] = manager_class(
                        name,
                        base_model,
                        system_prompt,
                        create_resource_cost,
                        invoke_resource_cost,
                        create_expense_cost,
                        invoke_expense_cost,
                        output_expense_cost
                    )
        except Exception as e:
            output_assistant_response(f"Error loading agents: {e}")
