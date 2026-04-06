from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, Tuple, List, Union
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
from src.metrics.semantic_metrics import compute_semantic_metrics_both
from src.manager.worker_model_policy import assert_base_model_allowed, is_base_model_allowed
from src.manager.orchestration_trace import (
    log_orchestration_event,
    worker_invocation_reprompt_flags,
)
from src.manager.semantic_ablation import (
    SEMANTIC_DENSITY_THRESHOLD,
    SEMANTIC_ENTROPY_THRESHOLD,
    apply_ablation_to_metrics,
    ceo_semantic_quality_followup,
    flags_dict,
    reprompt_triggered,
    semantic_metrics_sampling_enabled,
)

MODEL_PATH = "./src/models/"
MODEL_FILE_PATH = "./src/models/models.json"

# Extra stochastic completions per worker turn (same prompt) so entropy/density have multiple hypotheses.
NUM_EXTRA_RESPONSES_FOR_SEMANTIC = 4  # paper-style: 4+ samples; failures use `continue` so others still run.


def semantic_sample_temperature() -> Optional[float]:
    """
    Sampling temperature for *extra* completions used only for semantic entropy/density.
    Primary worker answers keep the model default so benchmarks stay stable.
    Set HASHIRU_SEMANTIC_SAMPLE_TEMPERATURE empty or \"default\" to omit (Ollama default).
    """
    raw = os.getenv("HASHIRU_SEMANTIC_SAMPLE_TEMPERATURE", "0.85").strip()
    if not raw or raw.lower() in ("default", "none", "off"):
        return None
    try:
        return float(raw)
    except ValueError:
        return 0.85


def _coerce_agent_reply(result: Any) -> tuple[str, Optional[float]]:
    """
    Normalize agent ask_agent return: plain str or dict with text/content + optional sequence_logprob.
    sequence_logprob: sum of chosen-token logprobs (natural log) when returned by Ollama logprob path.
    """
    if isinstance(result, dict):
        text = result.get("text")
        if text is None:
            text = result.get("content", "")
        text = (text or "").strip()
        sl = result.get("sequence_logprob")
        try:
            seq_f = float(sl) if sl is not None else None
        except (TypeError, ValueError):
            seq_f = None
        return text, seq_f
    return str(result or "").strip(), None


def compact_worker_reply_for_ceo(result: Any) -> Any:
    """
    Strip per-token logprobs from worker replies before they go to the CEO / tool JSON.
    Full structures are huge and blow Gemini context (and Gradio tool preview).
    """
    if not isinstance(result, dict):
        return result
    if "token_logprobs" not in result and "token_logprob_steps" not in result:
        return result
    compact: dict[str, Any] = {}
    if result.get("text") is not None:
        compact["text"] = result["text"]
    elif result.get("content") is not None:
        compact["text"] = result["content"]
    if result.get("sequence_logprob") is not None:
        compact["sequence_logprob"] = result["sequence_logprob"]
    if result.get("token_logprob_steps") is not None:
        compact["token_logprob_steps"] = result["token_logprob_steps"]
    return compact


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
    def ask_agent(self, prompt: str, **kwargs) -> Any:
        """ask agent a question (optional kwargs e.g. temperature for local models)"""
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

    def ask_agent(self, prompt, **kwargs) -> Union[str, dict]:
        output_assistant_response(f"Asked Agent {self.agent_name} a question")
        from src.manager.ollama_logprobs import (
            ollama_chat_with_logprobs,
            ollama_logprobs_feature_enabled,
        )

        temperature = kwargs.get("temperature")

        # Try Ollama logprobs for any local Ollama model. Not all models/builds expose it;
        # failures are handled by the fallback chat path below.
        if ollama_logprobs_feature_enabled():
            try:
                text, seq_lp, raw_lp, _data = ollama_chat_with_logprobs(
                    model=self.agent_name,
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    temperature=temperature,
                )
                output_assistant_response(
                    f"Agent {self.agent_name} answered with {text}"
                )
                n_tok = len(raw_lp) if raw_lp else 0
                return {
                    "text": text,
                    "sequence_logprob": seq_lp,
                    "token_logprobs": raw_lp,
                    "token_logprob_steps": n_tok,
                }
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Ollama logprob chat failed for %s, falling back to library chat: %s",
                    self.agent_name,
                    e,
                )

        chat_kwargs: dict[str, Any] = {
            "model": self.agent_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            chat_kwargs["options"] = {"temperature": float(temperature)}
        agent_response = ollama.chat(**chat_kwargs)
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

    def ask_agent(self, prompt, **kwargs):
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

    def ask_agent(self, prompt: str, **kwargs) -> str:
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

    def ask_agent(self, prompt: str, **kwargs) -> str:
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

class OpenAIAgent(Agent):
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
        output_expense_cost: int = 0,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required for ChatGPT models. "
                "Set OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=self.api_key)
        super().__init__(
            agent_name,
            base_model,
            system_prompt,
            create_resource_cost,
            invoke_resource_cost,
            create_expense_cost,
            invoke_expense_cost,
            output_expense_cost,
        )

    def create_model(self) -> None:
        # OpenAI-hosted models do not require local creation.
        pass

    @staticmethod
    def _sum_choice_logprobs(choice: Any) -> Optional[float]:
        """
        Sum chosen-token logprobs when the provider returns them.
        Returns None when unavailable.
        """
        try:
            lp_obj = getattr(choice, "logprobs", None)
            content = getattr(lp_obj, "content", None)
            if not content:
                return None
            total = 0.0
            n = 0
            for step in content:
                lp = getattr(step, "logprob", None)
                if lp is None:
                    continue
                total += float(lp)
                n += 1
            return total if n > 0 else None
        except Exception:
            return None

    def ask_agent(self, prompt: str, **kwargs) -> Union[str, dict]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.base_model,
                messages=messages,
                logprobs=True,
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            seq_lp = self._sum_choice_logprobs(choice)
            return {
                "text": text,
                "sequence_logprob": seq_lp,
            }
        except Exception as e:
            logging.getLogger(__name__).warning(
                "OpenAI logprob chat failed for %s, falling back without logprobs: %s",
                self.base_model,
                e,
            )

        response = self.client.chat.completions.create(
            model=self.base_model,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

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
        self._agent_histories: Dict[str, List[dict]] = {}
        self.logger = logging.getLogger("AgentManager")
        self._agent_types = {
            "ollama": OllamaAgent,
            # @@ something weird is going on with the google api key, disable gemini agents for now
            # "gemini": GeminiAgent,
            "groq": GroqAgent,
            "lambda": LambdaAgent,
            "openai": OpenAIAgent,
        }

        self._load_agents()

    def get_agent_history(self, agent_name: str) -> List[dict]:
        return list(self._agent_histories.get(agent_name, []))

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

        assert_base_model_allowed(base_model)

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

    def get_agent_responses(
        self, agent_name: str, prompt: str, num_responses: int
    ) -> Tuple[list, list]:
        """
        Get num_responses from the agent for the same prompt (for semantic metrics).
        Returns (texts, sequence_logprobs): parallel lists (one float or None per text
        when the backend returns sequence_logprob). Does not compute metrics or append to history.
        """
        agent = self.get_agent(agent_name)
        if not self.is_local_invocation_enabled and agent.get_type() == "local":
            raise ValueError("Local invocation mode is disabled.")
        if not self.is_cloud_invocation_enabled and agent.get_type() == "cloud":
            raise ValueError("Cloud invocation mode is disabled.")
        texts: list[str] = []
        seq_lps: list[Optional[float]] = []
        n_tokens_prompt = len(prompt.split()) / 1000000
        ss_temp = semantic_sample_temperature()
        for _ in range(num_responses):
            try:
                self.validate_budget(agent.invoke_resource_cost,
                                     agent.invoke_expense_cost * n_tokens_prompt)
                self.budget_manager.add_to_expense_budget(agent.invoke_expense_cost * n_tokens_prompt)
                result = agent.ask_agent(prompt, temperature=ss_temp)
                t, lp = _coerce_agent_reply(result)
                if t and str(t).strip():
                    texts.append(str(t).strip())
                    seq_lps.append(lp)
                n_out = len((t or "").split()) / 1000000
                self.budget_manager.add_to_expense_budget(agent.output_expense_cost * n_out)
            except Exception as e:
                self.logger.warning("get_agent_responses: one call failed (continuing): %s", e)
                continue
        return texts, seq_lps

    def list_agents(self) -> dict:
        """Return agent information (name, description, costs)"""
        try:
            if os.path.exists(MODEL_FILE_PATH):
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    full_models = json.loads(f.read())

                # Include description/specialty so CEO can judge overlap before creating new agents
                simplified_agents = {}
                for name, data in full_models.items():
                    bm = data.get("base_model", "")
                    if not is_base_model_allowed(bm):
                        continue
                    desc = data.get("description", "")
                    simplified_agents[name] = {
                        "description": desc,
                        "specialty": desc,
                        "create_resource_cost": data.get("create_resource_cost", 0),
                        "invoke_resource_cost": data.get("invoke_resource_cost", 0),
                        "create_expense_cost": data.get("create_expense_cost", 0),
                        "invoke_expense_cost": data.get("invoke_expense_cost", 0),
                        "base_model": bm,
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

    def ask_agent(self, agent_name: str, prompt: str, **kwargs):
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
        text, primary_seq_lp = _coerce_agent_reply(result)
        n_tokens = len((text or "").split()) / 1000000
        self.budget_manager.add_to_expense_budget(
            agent.output_expense_cost * n_tokens)

        # --- Gather extra responses for semantic metrics (same prompt, multiple samples) ---
        samples_for_metrics: list[str] = []
        sample_seq_lps: list[Optional[float]] = []
        ss_temp = semantic_sample_temperature()
        if semantic_metrics_sampling_enabled():
            for _ in range(NUM_EXTRA_RESPONSES_FOR_SEMANTIC):
                try:
                    self.validate_budget(agent.invoke_resource_cost,
                                         agent.invoke_expense_cost * n_tokens)
                    self.budget_manager.add_to_expense_budget(
                        agent.invoke_expense_cost * n_tokens)
                    extra_result = agent.ask_agent(prompt, temperature=ss_temp)
                    extra_text, extra_lp = _coerce_agent_reply(extra_result)
                    if extra_text and extra_text.strip():
                        samples_for_metrics.append(extra_text.strip())
                        sample_seq_lps.append(extra_lp)
                    n_tokens_out = len((extra_text or "").split()) / 1000000
                    self.budget_manager.add_to_expense_budget(
                        agent.output_expense_cost * n_tokens_out)
                except Exception as e:
                    self.logger.warning(
                        "Extra response for semantic metrics failed (will try remaining samples): %s", e
                    )
                    continue

        # --- Always one gateway call per worker turn when semantic metrics are enabled (may be 0 extra samples). ---
        entropy, density = None, None
        both_diag = None
        sequence_logprobs: Optional[list] = None
        if semantic_metrics_sampling_enabled():
            if samples_for_metrics:
                if primary_seq_lp is not None or any(
                    x is not None for x in sample_seq_lps
                ):
                    sequence_logprobs = [primary_seq_lp] + sample_seq_lps
            elif primary_seq_lp is not None:
                sequence_logprobs = [primary_seq_lp]
            try:
                both = compute_semantic_metrics_both(
                    prompt=prompt,
                    response=text,
                    samples=samples_for_metrics if samples_for_metrics else None,
                    sequence_logprobs=sequence_logprobs,
                )
                entropy, density = apply_ablation_to_metrics(
                    both.get("entropy"), both.get("density")
                )
                both_diag = both.get("diagnostics")
                if not both.get("diagnostics", {}).get("entropy_ok"):
                    self.logger.debug("Entropy backend reported error: %s", both.get("diagnostics", {}).get("entropy_error"))
                if not both.get("diagnostics", {}).get("density_ok"):
                    self.logger.debug("Density backend reported error: %s", both.get("diagnostics", {}).get("density_error"))
            except Exception as e:
                self.logger.warning("Semantic metric compute failed for agent %s: %s", agent_name, e)
            self.logger.info(
                "Semantic metrics (%s): samples_collected=%s entropy=%s density=%s entropy_ok=%s density_ok=%s",
                agent_name,
                len(samples_for_metrics),
                entropy,
                density,
                (both_diag or {}).get("entropy_ok"),
                (both_diag or {}).get("density_ok"),
            )

        prompt_excerpt = prompt
        if isinstance(prompt, str) and len(prompt) > 500:
            prompt_excerpt = prompt[:500] + "..."

        entry = {
            "role": "agent",
            "text": text,
            "semantic_entropy": entropy,
            "semantic_density": density,
            "semantic_quality_concern": reprompt_triggered(entropy, density),
            "timestamp": time.time(),
            "prompt_excerpt": prompt_excerpt,
        }
        try:
            append_fn = getattr(agent, "append_to_history", None)
            if callable(append_fn):
                append_fn(entry)
            else:
                try:
                    self._agent_histories.setdefault(agent_name, []).append(entry)
                except Exception:
                    self.logger.warning("Could not append agent history for %s", getattr(agent, "name", agent_name))
        except Exception as e:
            self.logger.exception("append_to_history invocation failed: %s", e)

        ceo_followup = ceo_semantic_quality_followup(
            agent_name=agent_name,
            worker_prompt=prompt,
            entropy=entropy,
            density=density,
        )
        implied_hint = (
            "semantic_metrics_below_bar_ceo_should_re_evaluate_or_reprompt"
            if ceo_followup["semantic_quality_concern"]
            else "semantic_metrics_acceptable_ceo_may_use_or_finalize"
        )
        n_aux = len(samples_for_metrics)
        # CEO / AskAgent tool output always uses the first forward pass only; extras are metrics-only.
        meta = {
            "agent_name": agent_name,
            "base_model": getattr(agent, "base_model", None),
            "worker_prompt": prompt,
            "worker_response": text,
            "semantic_entropy": entropy,
            "semantic_density": density,
            "semantic_entropy_threshold": SEMANTIC_ENTROPY_THRESHOLD,
            "semantic_density_threshold": SEMANTIC_DENSITY_THRESHOLD,
            "num_stochastic_samples_first_round": n_aux,
            "metrics_diagnostics_final": both_diag,
            "semantic_ablation": flags_dict(),
            "sequence_logprobs": sequence_logprobs,
            "semantic_quality_concern": ceo_followup["semantic_quality_concern"],
            "semantic_threshold_violations": ceo_followup["semantic_threshold_violations"],
            "semantic_quality_summary": ceo_followup["semantic_quality_summary"],
            "suggested_ceo_next_steps": ceo_followup["suggested_ceo_next_steps"],
            "worker_prompt_excerpt_for_ceo": ceo_followup["worker_prompt_excerpt_for_ceo"],
            "implied_ceo_decision_hint": implied_hint,
            "worker_response_kind": "primary",
            "semantic_auxiliary_completions_count": n_aux,
            "semantic_metrics_include_auxiliary_samples": bool(samples_for_metrics),
            "same_prompt_completion_index_sent_to_ceo": 0,
            "same_prompt_total_llm_completions": 1 + n_aux,
        }
        inv_idx, reprompted = worker_invocation_reprompt_flags(
            agent_name, bool(meta["semantic_quality_concern"])
        )
        meta["worker_invocation_index"] = inv_idx
        meta["worker_reprompted_after_semantic_check"] = reprompted
        log_orchestration_event(
            "worker_answer",
            worker_routing="AskAgent",
            phase="worker_completion",
            agent_name=agent_name,
            base_model=meta["base_model"],
            worker_prompt=prompt,
            worker_response=text,
            worker_response_kind="primary",
            semantic_auxiliary_completions_count=n_aux,
            semantic_metrics_include_auxiliary_samples=bool(samples_for_metrics),
            same_prompt_completion_index_sent_to_ceo=0,
            same_prompt_total_llm_completions=1 + n_aux,
            semantic_entropy=entropy,
            semantic_density=density,
            semantic_entropy_threshold=SEMANTIC_ENTROPY_THRESHOLD,
            semantic_density_threshold=SEMANTIC_DENSITY_THRESHOLD,
            worker_invocation_index=inv_idx,
            worker_reprompted_after_semantic_check=reprompted,
            semantic_quality_concern=meta["semantic_quality_concern"],
            semantic_threshold_violations=meta["semantic_threshold_violations"],
            semantic_quality_summary=meta["semantic_quality_summary"],
            suggested_ceo_next_steps=meta["suggested_ceo_next_steps"],
            implied_ceo_decision_hint=implied_hint,
        )

        return (
            result,
            self.budget_manager.get_current_remaining_resource_budget(),
            self.budget_manager.get_current_remaining_expense_budget(),
            meta,
        )

    def ask_multiple_agents(
        self,
        agent_prompts: list,
        user_question: str = "",
        metric_scope: str = "per_agent",
        **kwargs
    ) -> Tuple[dict, Optional[float], Optional[float], int, int]:
        """
        Ask multiple agents (each with its own prompt), then compute semantic metrics.
        Default behavior is per-agent metrics only (metric_scope="per_agent"), which is
        safer for heterogeneous sub-tasks. Optional metric_scope="global" computes one
        aggregate metric over all responses for same-question, multi-viewpoint scenarios.

        agent_prompts: list of dicts with keys "agent_name" and "prompt"
        user_question: overall question (used for metrics); if empty, first prompt is used.

        Returns: (result_dict, semantic_entropy, semantic_density, resource_budget, expense_budget)
        result_dict has "combined_output", "per_agent_outputs", "semantic_entropy", "semantic_density".
        """
        if not agent_prompts:
            raise ValueError("agent_prompts must be a non-empty list of {agent_name, prompt}")
        metric_scope = (metric_scope or "per_agent").strip().lower()
        if metric_scope not in ("per_agent", "global"):
            raise ValueError("metric_scope must be one of: per_agent, global")
        primaries = []
        all_individual_responses = []
        all_seq_lps: list[Optional[float]] = []
        per_agent_outputs = []

        for item in agent_prompts:
            name = item.get("agent_name")
            prompt = item.get("prompt", "")
            if not name or not prompt:
                continue
            agent = self.get_agent(name)
            if not self.is_local_invocation_enabled and agent.get_type() == "local":
                raise ValueError("Local invocation mode is disabled.")
            if not self.is_cloud_invocation_enabled and agent.get_type() == "cloud":
                raise ValueError("Cloud invocation mode is disabled.")
            n_tokens = len(prompt.split()) / 1000000
            self.validate_budget(agent.invoke_resource_cost, agent.invoke_expense_cost * n_tokens)
            self.budget_manager.add_to_expense_budget(agent.invoke_expense_cost * n_tokens)
            result = agent.ask_agent(prompt)
            text, primary_lp = _coerce_agent_reply(result)
            text = (text or "").strip()
            n_out = len((text or "").split()) / 1000000
            self.budget_manager.add_to_expense_budget(agent.output_expense_cost * n_out)
            primaries.append((name, text))
            all_individual_responses.append(text)
            all_seq_lps.append(primary_lp)

            extra: list[str] = []
            extra_lps: list[Optional[float]] = []
            if semantic_metrics_sampling_enabled():
                extra, extra_lps = self.get_agent_responses(
                    name, prompt, NUM_EXTRA_RESPONSES_FOR_SEMANTIC
                )
                all_individual_responses.extend(extra)
                all_seq_lps.extend(extra_lps)

            pe, pd = None, None
            pediag = None
            seq_i: Optional[list] = None
            if semantic_metrics_sampling_enabled() and text:
                if primary_lp is not None or any(
                    x is not None for x in extra_lps
                ):
                    seq_i = [primary_lp] + extra_lps
                try:
                    both_i = compute_semantic_metrics_both(
                        prompt=prompt,
                        response=text,
                        samples=extra,
                        sequence_logprobs=seq_i,
                    )
                    pe, pd = apply_ablation_to_metrics(
                        both_i.get("entropy"), both_i.get("density")
                    )
                    pediag = both_i.get("diagnostics")
                except Exception as e:
                    self.logger.warning(
                        "Per-agent semantic metrics failed for %s: %s", name, e
                    )
            concern_i = bool(reprompt_triggered(pe, pd))
            inv_m, rep_m = worker_invocation_reprompt_flags(name, concern_i)
            n_aux_m = len(extra)
            per_agent_outputs.append(
                {
                    "agent_name": name,
                    "prompt": prompt,
                    "response": text,
                    "worker_response_kind": "primary",
                    "semantic_auxiliary_completions_count": n_aux_m,
                    "semantic_metrics_include_auxiliary_samples": bool(extra),
                    "same_prompt_completion_index_sent_to_ceo": 0,
                    "same_prompt_total_llm_completions": 1 + n_aux_m,
                    "base_model": getattr(agent, "base_model", None),
                    "semantic_entropy": pe,
                    "semantic_density": pd,
                    "num_stochastic_samples": n_aux_m,
                    "metrics_diagnostics": pediag,
                    "semantic_ablation": flags_dict(),
                    "sequence_logprobs": seq_i,
                    "semantic_quality_concern": concern_i,
                    "worker_invocation_index": inv_m,
                    "worker_reprompted_after_semantic_check": rep_m,
                }
            )
            multi_hint = (
                "semantic_metrics_below_bar_ceo_should_re_evaluate_or_reprompt"
                if concern_i
                else "semantic_metrics_acceptable_ceo_may_use_or_finalize"
            )
            log_orchestration_event(
                "worker_answer_multi",
                worker_routing="AskMultipleAgents",
                phase="worker_completion",
                agent_name=name,
                base_model=getattr(agent, "base_model", None),
                worker_prompt=prompt,
                worker_response=text,
                worker_response_kind="primary",
                semantic_auxiliary_completions_count=n_aux_m,
                semantic_metrics_include_auxiliary_samples=bool(extra),
                same_prompt_completion_index_sent_to_ceo=0,
                same_prompt_total_llm_completions=1 + n_aux_m,
                user_question=user_question or None,
                semantic_entropy=pe,
                semantic_density=pd,
                semantic_entropy_threshold=SEMANTIC_ENTROPY_THRESHOLD,
                semantic_density_threshold=SEMANTIC_DENSITY_THRESHOLD,
                semantic_quality_concern=concern_i,
                worker_invocation_index=inv_m,
                worker_reprompted_after_semantic_check=rep_m,
                implied_ceo_decision_hint=multi_hint,
            )

        combined_response = "\n\n".join(
            f"**{name}:**\n{text}" for name, text in primaries
        )
        metrics_prompt = user_question.strip() or (agent_prompts[0].get("prompt", ""))

        entropy, density = None, None
        seq_combined: Optional[list] = None

        if metric_scope == "global":
            # Primary `combined_response` is not a single model run — no sequence LP;
            # align [None] + per-utterance LPs with samples.
            if semantic_metrics_sampling_enabled() and any(
                x is not None for x in all_seq_lps
            ):
                seq_combined = [None] + list(all_seq_lps)
            if (
                semantic_metrics_sampling_enabled()
                and len(all_individual_responses) >= 1
            ):
                try:
                    both = compute_semantic_metrics_both(
                        prompt=metrics_prompt,
                        response=combined_response,
                        samples=all_individual_responses,
                        sequence_logprobs=seq_combined,
                    )
                    entropy, density = apply_ablation_to_metrics(
                        both.get("entropy"), both.get("density")
                    )
                except Exception as e:
                    self.logger.warning("Semantic metric compute failed for multi-agent: %s", e)
        else:
            # Default: summarize per-agent metrics to avoid false concern on heterogeneous
            # sub-tasks that naturally produce diverse outputs.
            pe_vals = [
                row.get("semantic_entropy")
                for row in per_agent_outputs
                if row.get("semantic_entropy") is not None
            ]
            pd_vals = [
                row.get("semantic_density")
                for row in per_agent_outputs
                if row.get("semantic_density") is not None
            ]
            entropy = (sum(pe_vals) / len(pe_vals)) if pe_vals else None
            density = (sum(pd_vals) / len(pd_vals)) if pd_vals else None

        result_dict = {
            "text": combined_response,
            "combined_output": combined_response,
            "per_agent_outputs": per_agent_outputs,
            "semantic_entropy": entropy,
            "semantic_density": density,
            "semantic_metric_scope": metric_scope,
            "semantic_ablation": flags_dict(),
            "sequence_logprobs": seq_combined,
        }
        log_orchestration_event(
            "worker_answers_multi_combined",
            worker_routing="AskMultipleAgents",
            phase="multi_agent_round_summary",
            n_agents=len(per_agent_outputs),
            user_question=metrics_prompt,
            semantic_metric_scope=metric_scope,
            semantic_entropy=entropy,
            semantic_density=density,
            semantic_entropy_threshold=SEMANTIC_ENTROPY_THRESHOLD,
            semantic_density_threshold=SEMANTIC_DENSITY_THRESHOLD,
            per_agent_outputs=per_agent_outputs,
        )
        return (
            result_dict,
            entropy,
            density,
            self.budget_manager.get_current_remaining_resource_budget(),
            self.budget_manager.get_current_remaining_expense_budget(),
        )

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
        model_key = (base_model or "").strip().lower()
        if model_key == "llama3.2":
            return "ollama"
        elif model_key == "mistral":
            return "ollama"
        elif model_key == "deepseek-r1":
            return "ollama"
        elif model_key in {"chatgpt-5.4", "gpt-5.4"} or "chatgpt" in model_key:
            return "openai"
        elif "gemini" in model_key:
            return "gemini"
        elif "groq" in model_key:
            return "groq"
        elif model_key.startswith("lambda-"):
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
                if not is_base_model_allowed(base_model):
                    self.logger.info(
                        "Skipping persisted agent %r: base_model %r not allowed by worker model policy",
                        name,
                        base_model,
                    )
                    continue
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
