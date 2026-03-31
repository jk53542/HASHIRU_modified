from enum import Enum, auto
from typing import List
from google import genai
from google.genai import types
from google.genai.types import *
import ast
import os
from dotenv import load_dotenv
import sys
from src.manager.agent_manager import AgentManager
from src.manager.budget_manager import BudgetManager
from src.manager.tool_manager import ToolManager
from src.manager.utils.suppress_outputs import suppress_output
import logging
import gradio as gr
from sentence_transformers import SentenceTransformer
import torch
from src.tools.default_tools.memory_manager import MemoryManager
from pathlib import Path
from google.genai.errors import APIError
import backoff
import mimetypes
import json
import traceback
from src.manager.orchestration_trace import log_orchestration_event

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)

# Gradio 5 walks message content and treats nested dict/list values like file paths.
# Prefix + JSON keeps tool/function_call payloads as plain strings (no path traversal).
INTERNAL_TOOL_JSON_PREFIX = "hashiru-internal-json:"

_WORKER_TOOLS = frozenset({"AskAgent", "AskMultipleAgents"})


def _encode_internal_tool_payload(obj) -> str:
    return INTERNAL_TOOL_JSON_PREFIX + json.dumps(obj, default=str, ensure_ascii=False)


def _decode_internal_tool_payload(content):
    """Return list/dict payload for tool/function_call roles, or None if not our format."""
    if isinstance(content, list):
        return content
    if not isinstance(content, str) or not content:
        return None
    if content.startswith(INTERNAL_TOOL_JSON_PREFIX):
        try:
            return json.loads(content[len(INTERNAL_TOOL_JSON_PREFIX) :])
        except json.JSONDecodeError:
            return None
    return None


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _index_last_user_with_agent_mandate(messages) -> int:
    """Index of the latest user message that requires worker delegation (benchmark convention)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        c = messages[i].get("content", "")
        if not isinstance(c, str):
            continue
        if "IMPORTANT CEO INSTRUCTIONS" in c or "You MUST use agents" in c:
            return i
    return -1


def _messages_segment_for_worker_check(messages) -> list:
    """
    Messages after the user turn that mandates agents, or after last user if HASHIRU_REQUIRE_ASK_AGENT.
    """
    idx = _index_last_user_with_agent_mandate(messages)
    if idx >= 0:
        return messages[idx + 1 :]
    if _env_truthy("HASHIRU_REQUIRE_ASK_AGENT"):
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return messages[i + 1 :]
    return []


def _segment_used_worker_agent(segment: list) -> bool:
    for m in segment:
        if m.get("role") == "function_call":
            dec = _decode_internal_tool_payload(m.get("content"))
            if isinstance(dec, list):
                for item in dec:
                    if (
                        item.get("kind") == "function_call"
                        and item.get("name") in _WORKER_TOOLS
                    ):
                        return True
        if m.get("role") == "tool":
            dec = _decode_internal_tool_payload(m.get("content"))
            if isinstance(dec, list):
                for item in dec:
                    if (
                        item.get("kind") == "function_response"
                        and item.get("name") in _WORKER_TOOLS
                    ):
                        return True
    return False


def _mandate_worker_enforcement_active(messages) -> bool:
    if _index_last_user_with_agent_mandate(messages) >= 0:
        return True
    return _env_truthy("HASHIRU_REQUIRE_ASK_AGENT")


# handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Mode(Enum):
    ENABLE_AGENT_CREATION = auto()
    ENABLE_LOCAL_AGENTS = auto()
    ENABLE_CLOUD_AGENTS = auto()
    ENABLE_TOOL_CREATION = auto()
    ENABLE_TOOL_INVOCATION = auto()
    ENABLE_RESOURCE_BUDGET = auto()
    ENABLE_ECONOMY_BUDGET = auto()
    ENABLE_MEMORY = auto()


def format_tool_response(response, indent=2):
    return json.dumps(response, indent=indent, ensure_ascii=False)


class GeminiManager:
    def __init__(self, system_prompt_file="./src/models/system6.prompt",
                 gemini_model="gemini-2.5-pro-exp-03-25",
                 modes: List[Mode] = []):
        self.input_tokens = 0
        self.output_tokens = 0
        self.logger = logging.getLogger("GeminiManager")
        load_dotenv()
        self.budget_manager = BudgetManager()

        self.toolsLoader: ToolManager = ToolManager()

        self.agentManager: AgentManager = AgentManager()

        self.API_KEY = os.getenv("GEMINI_KEY")
        self.client = genai.Client(api_key=self.API_KEY)
        self.model_name = gemini_model
        self.memory_manager = MemoryManager()
        with open(system_prompt_file, 'r', encoding="utf8") as f:
            self.system_prompt = f.read()
        self.messages = []
        self.max_tool_rounds = int(os.getenv("HASHIRU_MAX_TOOL_ROUNDS", "12"))
        self.max_same_tool_call_repeats = int(
            os.getenv("HASHIRU_MAX_SAME_TOOL_CALL_REPEATS", "4")
        )
        self.set_modes(modes)
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

    def get_current_modes(self):
        return [mode.name for mode in self.modes]

    def set_modes(self, modes: List[Mode]):
        self.modes = modes
        self.budget_manager.set_resource_budget_status(
            self.check_mode(Mode.ENABLE_RESOURCE_BUDGET))
        self.budget_manager.set_expense_budget_status(
            self.check_mode(Mode.ENABLE_ECONOMY_BUDGET))
        self.toolsLoader.set_creation_mode(
            self.check_mode(Mode.ENABLE_TOOL_CREATION))
        self.toolsLoader.set_invocation_mode(
            self.check_mode(Mode.ENABLE_TOOL_INVOCATION))
        self.agentManager.set_creation_mode(
            self.check_mode(Mode.ENABLE_AGENT_CREATION))
        self.agentManager.set_local_invocation_mode(
            self.check_mode(Mode.ENABLE_LOCAL_AGENTS))
        self.agentManager.set_cloud_invocation_mode(
            self.check_mode(Mode.ENABLE_CLOUD_AGENTS))

    def check_mode(self, mode: Mode):
        return mode in self.modes

    @backoff.on_exception(backoff.expo,
                          APIError,
                          max_tries=3,
                          jitter=None)
    def generate_response(self, messages):
        tools = self.toolsLoader.getTools()
        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=messages,
        )
        self.budget_manager.add_to_expense_budget(
            response.total_tokens * 0.10/1000000  # Assuming $0.10 per million tokens
        )
        self.input_tokens += response.total_tokens
        return self.client.models.generate_content_stream(
            model=self.model_name,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,
                tools=tools,
                safety_settings=self.safety_settings,
            ),
        )

    def handle_tool_calls(self, function_calls):
        parts = []
        i = 0
        for function_call in function_calls:
            title = ""
            thinking = ""
            toolResponse = None
            logger.info(
                f"Function Name: {function_call.name}, Arguments: {function_call.args}")
            title = f"Invoking `{function_call.name}` with \n```json\n{format_tool_response(function_call.args)}\n```\n"
            yield {
                "role": "assistant",
                "content": thinking,
                "metadata": {
                    "title": title,
                    "id": i,
                    "status": "pending",
                }
            }
            try:
                self.input_tokens += len(repr(function_call).split())
                toolResponse = self.toolsLoader.runTool(
                    function_call.name, function_call.args)
            except Exception as e:
                logger.warning(f"Error running tool: {e}")
                toolResponse = {
                    "status": "error",
                    "message": f"Tool `{function_call.name}` failed to run.",
                    "output": str(e),
                }
            try:
                if isinstance(toolResponse, dict):
                    trace_extras: dict = {}
                    if function_call.name == "AskAgent":
                        # Explicitly trace which worker was called for easier downstream analysis.
                        called_name = None
                        try:
                            called_name = (function_call.args or {}).get("agent_name")
                        except Exception:
                            called_name = None
                        if called_name:
                            trace_extras["called_agent_names"] = [called_name]
                            trace_extras["called_agent_count"] = 1
                        om = toolResponse.get("orchestration_meta")
                        if isinstance(om, dict):
                            trace_extras["agent_name"] = om.get("agent_name")
                            trace_extras["worker_prompt"] = om.get("worker_prompt")
                            trace_extras["semantic_entropy"] = om.get(
                                "semantic_entropy"
                            )
                            trace_extras["semantic_density"] = om.get(
                                "semantic_density"
                            )
                            trace_extras["worker_reprompted_after_semantic_check"] = (
                                om.get("worker_reprompted_after_semantic_check")
                            )
                            trace_extras["semantic_quality_concern"] = om.get(
                                "semantic_quality_concern"
                            )
                    if function_call.name == "AskMultipleAgents":
                        # Track agent list explicitly (not only nested in per_agent_outputs).
                        called_names = []
                        try:
                            raw_ap = (function_call.args or {}).get("agent_prompts_json")
                            if isinstance(raw_ap, str) and raw_ap.strip():
                                parsed_ap = json.loads(raw_ap)
                                if isinstance(parsed_ap, list):
                                    for item in parsed_ap:
                                        if isinstance(item, dict):
                                            n = item.get("agent_name")
                                            if isinstance(n, str) and n.strip():
                                                called_names.append(n.strip())
                        except Exception:
                            pass
                        if not called_names:
                            try:
                                pa = toolResponse.get("per_agent_outputs") or []
                                if isinstance(pa, list):
                                    for item in pa:
                                        if isinstance(item, dict):
                                            n = item.get("agent_name")
                                            if isinstance(n, str) and n.strip():
                                                called_names.append(n.strip())
                            except Exception:
                                pass
                        if called_names:
                            # Preserve order while de-duplicating.
                            dedup = list(dict.fromkeys(called_names))
                            trace_extras["called_agent_names"] = dedup
                            trace_extras["called_agent_count"] = len(dedup)
                        if toolResponse.get("semantic_metric_scope") is not None:
                            trace_extras["semantic_metric_scope"] = toolResponse.get(
                                "semantic_metric_scope"
                            )
                        trace_extras["semantic_entropy"] = toolResponse.get(
                            "semantic_entropy"
                        )
                        trace_extras["semantic_density"] = toolResponse.get(
                            "semantic_density"
                        )
                        trace_extras["per_agent_outputs"] = toolResponse.get(
                            "per_agent_outputs"
                        )
                    log_orchestration_event(
                        "ceo_tool_finished",
                        tool=function_call.name,
                        status=toolResponse.get("status"),
                        message=toolResponse.get("message"),
                        args=function_call.args,
                        **trace_extras,
                    )
            except Exception:
                logger.debug("orchestration trace logging failed", exc_info=True)
            logger.debug(f"Tool Response: {toolResponse}")
            thinking += f"Tool responded with \n```json\n{format_tool_response(toolResponse)}\n```\n"
            yield {
                "role": "assistant",
                "content": thinking,
                "metadata": {
                    "title": title,
                    "id": i,
                    "status": "done",
                }
            }
            tool_content = types.Part.from_function_response(
                name=function_call.name,
                response={"result": toolResponse})
            try:
                if function_call.name == "ToolCreator" or function_call.name == "ToolDeletor":
                    self.toolsLoader.load_tools()
            except Exception as e:
                logger.info(
                    f"Error loading tools: {str(e)}. Deleting the tool.")
                yield {
                    "role": "assistant",
                    "content": f"Error loading tools: {str(e)}. Deleting the tool.\n",
                    "metadata": {
                        "title": "Trying to load the newly created tool",
                        "id": i,
                        "status": "done",
                    }
                }
                # delete the created tool
                self.toolsLoader.delete_tool(
                    toolResponse['output']['tool_name'], toolResponse['output']['tool_file_path'])
                tool_content = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": f"{function_call.name} with {function_call.args} doesn't follow the required format, please read the other tool implementations for reference." + str(e)})
            parts.append(tool_content)
            i += 1
        self.output_tokens += len(repr(parts).split())
        # Persist as plain data instead of repr(...) to avoid eval/parsing failures.
        payload = []
        for p in parts:
            try:
                if getattr(p, "function_response", None):
                    fr = p.function_response
                    payload.append({
                        "kind": "function_response",
                        "name": fr.name,
                        "response": fr.response,
                    })
            except Exception:
                continue
        yield {
            "role": "tool",
            "content": _encode_internal_tool_payload(payload),
        }

    def format_chat_history(self, messages=[]):
        formatted_history = []
        for message in messages:
            # Skip thinking messages (messages with metadata)
            if not ((message.get("role") == "assistant" and "metadata" in message
                     and message["metadata"] is not None)):
                role = "model"
                match message.get("role"):
                    case "user":
                        role = "user"
                        if isinstance(message["content"], tuple):
                            path = message["content"][0]
                            try:
                                image_bytes = open(path, "rb").read()
                                mime_type, _ = mimetypes.guess_type(path)
                                parts = [
                                    types.Part.from_bytes(
                                        data=image_bytes,
                                        mime_type=mime_type
                                    ),
                                ]
                            except Exception as e:
                                logger.error(f"Error uploading file: {e}")
                                parts = [types.Part.from_text(
                                    text="Error uploading file: "+str(e))]
                            formatted_history.append(
                                types.Content(
                                    role=role,
                                    parts=parts
                                ))
                            continue
                        else:
                            parts = [types.Part.from_text(
                                text=message.get("content", ""))]
                    case "memories":
                        role = "user"
                        parts = [types.Part.from_text(
                            text="Here are the relevant memories for the user's query: "+message.get("content", ""))]
                    case "tool":
                        role = "tool"
                        content = message.get("content", "")
                        # Decode Gradio-safe string payloads (see INTERNAL_TOOL_JSON_PREFIX).
                        decoded = _decode_internal_tool_payload(content)
                        if decoded is None and isinstance(content, str):
                            try:
                                maybe = json.loads(content)
                                if isinstance(maybe, list):
                                    decoded = maybe
                            except json.JSONDecodeError:
                                pass
                        if decoded is not None:
                            content = decoded
                        # List of serialized function responses.
                        if isinstance(content, list):
                            tool_parts = []
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("kind") != "function_response":
                                    continue
                                try:
                                    tool_parts.append(types.Part.from_function_response(
                                        name=item.get("name", ""),
                                        response=item.get("response", {})
                                    ))
                                except Exception:
                                    continue
                            if tool_parts:
                                formatted_history.append(types.Content(role="tool", parts=tool_parts))
                        # Backward compatibility with older saved repr(...) content.
                        elif isinstance(content, str):
                            try:
                                parsed = ast.literal_eval(content)
                                if isinstance(parsed, types.Content):
                                    formatted_history.append(parsed)
                            except Exception:
                                logger.warning("Skipping unparsable tool content in chat history.")
                        continue
                    case "function_call":
                        role = "model"
                        content = message.get("content", "")
                        decoded = _decode_internal_tool_payload(content)
                        if decoded is None and isinstance(content, str):
                            try:
                                maybe = json.loads(content)
                                if isinstance(maybe, list):
                                    decoded = maybe
                            except json.JSONDecodeError:
                                pass
                        if decoded is not None:
                            content = decoded
                        # Serialized function-call parts.
                        if isinstance(content, list):
                            call_parts = []
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("kind") != "function_call":
                                    continue
                                try:
                                    call_parts.append(types.Part.from_function_call(
                                        name=item.get("name", ""),
                                        args=item.get("args", {}) or {},
                                    ))
                                except Exception:
                                    continue
                            if call_parts:
                                formatted_history.append(types.Content(role="model", parts=call_parts))
                        # Backward compatibility with older saved repr(...) content.
                        elif isinstance(content, str):
                            try:
                                parsed = ast.literal_eval(content)
                                if isinstance(parsed, types.Content):
                                    formatted_history.append(parsed)
                            except Exception:
                                logger.warning("Skipping unparsable function_call content in chat history.")
                        continue
                    case _:
                        role = "model"
                        content = message.get("content", "")
                        if content.strip() == "":
                            print("Empty message received: ", message)
                            continue
                        parts = [types.Part.from_text(
                            text=content)]
                formatted_history.append(types.Content(
                    role=role,
                    parts=parts
                ))
        return formatted_history

    def get_k_memories(self, query, k=5, threshold=0.0):
        raw_memories = MemoryManager().get_memories()
        memories = []
        for i in range(len(raw_memories)):
            memories.append(raw_memories[i]['memory'])
        if len(memories) == 0:
            return []
        top_k = min(k, len(memories))
        # Semantic Retrieval with GPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        else:
            device = 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        doc_embeddings = model.encode(
            memories, convert_to_tensor=True, device=device)
        query_embedding = model.encode(
            query, convert_to_tensor=True, device=device)
        similarity_scores = model.similarity(
            query_embedding, doc_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)
        results = []
        for score, idx in zip(scores, indices):
            if score >= threshold:
                results.append(raw_memories[idx.item()])
        return results

    def run(self, messages):
        try:
            if self.check_mode(Mode.ENABLE_MEMORY) and len(messages) > 0:
                memories = self.get_k_memories(
                    messages[-1]['content'], k=5, threshold=0.1)
                if len(memories) > 0:
                    messages.append({
                        "role": "memories",
                        "content": f"{memories}",
                    })
                    messages.append({
                        "role": "assistant",
                        "content": f"Memories: \n```json\n{format_tool_response(memories)}\n```\n",
                        "metadata": {"title": "Memories"}
                    })
                    yield messages
        except Exception as e:
            pass
        yield from self.invoke_manager(messages, tool_round=0, tool_call_counts={})
        print("Tokens used: Input: {}, Output: {}".format(
            self.input_tokens, self.output_tokens))

    def _function_call_signature(self, function_call):
        try:
            args = function_call.args if function_call.args is not None else {}
            args_key = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            args_key = str(function_call.args)
        return f"{function_call.name}|{args_key}"

    def invoke_manager(self, messages, tool_round=0, tool_call_counts=None, mandate_nudges=0):
        if tool_call_counts is None:
            tool_call_counts = {}
        max_agent_mandate_nudges = int(
            os.environ.get("HASHIRU_MANDATE_ASK_AGENT_MAX_NUDGES", "4")
        )
        chat_history = self.format_chat_history(messages)
        logger.debug(f"Chat history: {chat_history}")
        try:
            response_stream = self.generate_response(chat_history)
            full_text = ""  # Accumulate the text from the stream
            function_calls = []
            function_call_requests = []
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    if full_text.strip() != "":
                        yield messages + [{
                            "role": "assistant",
                            "content": full_text
                        }]
                    else:
                        print("Empty chunk received")
                        print(chunk)
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        has_function_call = False
                        serialized_calls = []
                        for part in candidate.content.parts:
                            if part.function_call:
                                has_function_call = True
                                function_calls.append(part.function_call)
                                serialized_calls.append({
                                    "kind": "function_call",
                                    "name": part.function_call.name,
                                    "args": part.function_call.args,
                                })
                        if has_function_call:
                            function_call_requests.append({
                                "role": "function_call",
                                "content": _encode_internal_tool_payload(
                                    serialized_calls),
                            })
            if full_text.strip() != "":
                messages.append({
                    "role": "assistant",
                    "content": full_text,
                })
                self.output_tokens += len(full_text.split())
                self.budget_manager.add_to_expense_budget(
                    len(full_text.split()) * 0.40/1000000  # Assuming $0.40 per million tokens
                )
            if function_call_requests:
                messages = messages + function_call_requests
            yield messages
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print(messages)
            print(chat_history)
            messages.append({
                "role": "assistant",
                "content": f"Error generating response: {str(e)}",
                "metadata": {
                            "title": "Error generating response",
                            "id": 0,
                            "status": "done"
                            }
            })
            logger.error(f"Error generating response{e}")
            yield messages
            return messages

        # Check if any text was received
        if len(full_text.strip()) == 0 and len(function_calls) == 0:
            messages.append({
                "role": "assistant",
                "content": "No response from the model.",
                "metadata": {"title": "No response from the model."}
            })

        if function_calls and len(function_calls) > 0:
            if tool_round >= self.max_tool_rounds:
                messages.append({
                    "role": "assistant",
                    "content": (
                        "Stopping due to tool-loop guard: maximum tool rounds reached. "
                        "Please return a final answer without more tool calls."
                    ),
                    "metadata": {"title": "Tool-loop guard"},
                })
                yield messages
                return

            # Prevent infinite loops from repeating exactly the same tool invocation.
            repeated_signature = None
            for fc in function_calls:
                sig = self._function_call_signature(fc)
                new_count = tool_call_counts.get(sig, 0) + 1
                tool_call_counts[sig] = new_count
                if new_count > self.max_same_tool_call_repeats:
                    repeated_signature = sig
                    break
            if repeated_signature is not None:
                messages.append({
                    "role": "assistant",
                    "content": (
                        "Stopping due to repeated identical tool calls. "
                        "Please provide the final answer directly now."
                    ),
                    "metadata": {"title": "Repeated-tool-call guard"},
                })
                yield messages
                return

            for call in self.handle_tool_calls(function_calls):
                yield messages + [call]
                if (call.get("role") == "tool"
                        or (call.get("role") == "assistant" and call.get("metadata", {}).get("status") == "done")):
                    messages.append(call)
            yield from self.invoke_manager(
                messages,
                tool_round=tool_round + 1,
                tool_call_counts=tool_call_counts,
                mandate_nudges=mandate_nudges,
            )
        else:
            agents_enabled = self.check_mode(Mode.ENABLE_LOCAL_AGENTS) or self.check_mode(
                Mode.ENABLE_CLOUD_AGENTS
            )
            seg = _messages_segment_for_worker_check(messages)
            if (
                agents_enabled
                and _mandate_worker_enforcement_active(messages)
                and seg
                and full_text.strip()
                and not _segment_used_worker_agent(seg)
                and mandate_nudges < max_agent_mandate_nudges
            ):
                logger.info(
                    "Worker mandate: CEO produced text without AskAgent/AskMultipleAgents; nudging (n=%s/%s).",
                    mandate_nudges + 1,
                    max_agent_mandate_nudges,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM (enforcement): You replied without calling AskAgent or AskMultipleAgents. "
                        "This user turn requires delegating the question to at least one worker agent. "
                        "Call GetAgents if needed, then AskAgent or AskMultipleAgents, and base your final "
                        "answer on the worker output. Do not answer from the orchestrator model alone."
                    ),
                })
                yield messages
                yield from self.invoke_manager(
                    messages,
                    tool_round=tool_round,
                    tool_call_counts=tool_call_counts,
                    mandate_nudges=mandate_nudges + 1,
                )
                return
            yield messages
