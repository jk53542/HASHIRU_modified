"""
Ollama logprob helper with multi-endpoint fallback.

This module first tries Ollama native `/api/chat` with `logprobs`, then falls
back to Ollama's OpenAI-compatible `/v1/chat/completions` endpoint (also with
`logprobs`) when needed.

Env:
  OLLAMA_HOST   — default http://127.0.0.1:11434
  HASHIRU_OLLAMA_TOP_LOGPROBS — default 5 (top alternatives per token)
  HASHIRU_OLLAMA_CHAT_TIMEOUT — seconds for each logprob HTTP request (default 1200).
  HASHIRU_OLLAMA_LOGPROBS — set 0 to force-disable logprobs even when semantic metrics are on.
  HASHIRU_OLLAMA_SKIP_HTTP_LOGPROBS_REASONING — default 1: skip HTTP logprobs for known long-CoT
    Ollama bases (e.g. DeepSeek-R1); use ``ollama.chat`` instead. Extras still supply text samples.
  HASHIRU_OLLAMA_FORCE_HTTP_LOGPROBS — set 1 to always try HTTP logprobs when enabled.

Note: With semantic sampling off, ``OllamaAgent`` uses native ``ollama.chat`` only.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
TOP_LOGPROBS = int(os.getenv("HASHIRU_OLLAMA_TOP_LOGPROBS", "5"))
CHAT_TIMEOUT = float(os.getenv("HASHIRU_OLLAMA_CHAT_TIMEOUT", "1200"))


def ollama_logprobs_feature_enabled() -> bool:
    """Master switch; logprobs are only requested for open-weight Ollama bases when this is on."""
    v = os.getenv("HASHIRU_OLLAMA_LOGPROBS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def ollama_http_logprobs_viable_for_model(model_id: str) -> bool:
    """
    Use HTTP logprob chat for this model id. Long chain-of-thought models often exceed client
    timeouts on ``/api/chat`` with logprobs (Ollama 500 after ~CHAT_TIMEOUT seconds).
    """
    if os.getenv("HASHIRU_OLLAMA_FORCE_HTTP_LOGPROBS", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return True
    if os.getenv("HASHIRU_OLLAMA_SKIP_HTTP_LOGPROBS_REASONING", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return True
    m = (model_id or "").lower()
    if any(x in m for x in ("qwq", "o1-preview", "o1-mini", "deepseek-r1", "deepseek r1")):
        return False
    if "r1" in m and any(x in m for x in ("deepseek", "qwen", "llama", "dolphin")):
        return False
    return True


def sum_chosen_token_logprobs(logprobs: Any) -> float | None:
    """
    Sum log p(token_t | prefix) for each generated token (natural log).
    Ollama returns a list of objects with a `logprob` field for the sampled token.
    """
    if not logprobs or not isinstance(logprobs, list):
        return None
    total = 0.0
    n = 0
    for step in logprobs:
        if not isinstance(step, dict):
            continue
        lp = step.get("logprob")
        if lp is None:
            continue
        try:
            total += float(lp)
            n += 1
        except (TypeError, ValueError):
            continue
    return total if n > 0 else None


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(url, json=body, timeout=CHAT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON type from {url}: {type(data)}")
    return data


def get_ollama_server_version(host: str | None = None) -> str | None:
    """
    Best-effort version probe via /api/version.
    Returns version string (e.g., '0.12.11') or None if unavailable.
    """
    if requests is None:
        return None
    base = (host or OLLAMA_HOST).rstrip("/")
    url = f"{base}/api/version"
    try:
        data = _post_json(url, {})
        ver = data.get("version")
        return str(ver) if ver else None
    except Exception:
        return None


def _version_tuple(ver: str | None) -> tuple[int, ...]:
    if not ver:
        return tuple()
    nums = re.findall(r"\d+", ver)
    return tuple(int(x) for x in nums[:3])


def ollama_logprobs_likely_supported(host: str | None = None) -> bool | None:
    """
    Heuristic gate from article/release notes:
      logprobs in native/openai-compatible APIs are available in newer Ollama
      (notably >= 0.12.11 per article).
    Returns:
      True  -> likely supported
      False -> likely unsupported due to old version
      None  -> unknown (cannot detect)
    """
    ver = get_ollama_server_version(host=host)
    if not ver:
        return None
    return _version_tuple(ver) >= (0, 12, 11)


def _extract_native_chat_payload(data: dict[str, Any]) -> tuple[str, list | None]:
    msg = data.get("message") if isinstance(data, dict) else {}
    if isinstance(msg, dict):
        content = (msg.get("content") or "").strip() if msg.get("content") else ""
    else:
        content = ""

    logprobs = data.get("logprobs")
    if logprobs is None and isinstance(msg, dict):
        logprobs = msg.get("logprobs")
    return content, logprobs if isinstance(logprobs, list) else None


def _extract_openai_chat_payload(data: dict[str, Any]) -> tuple[str, list | None]:
    """
    Parse OpenAI-compatible chat completion response from Ollama:
      choices[0].message.content
      choices[0].logprobs.content[*].logprob
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
    content = (msg.get("content") or "").strip() if msg else ""

    lp_obj = c0.get("logprobs") if isinstance(c0, dict) else None
    lp_content = lp_obj.get("content") if isinstance(lp_obj, dict) else None
    if not isinstance(lp_content, list):
        return content, None

    normalized: list[dict[str, Any]] = []
    for item in lp_content:
        if not isinstance(item, dict):
            continue
        if "logprob" not in item:
            continue
        normalized.append(
            {
                "token": item.get("token"),
                "logprob": item.get("logprob"),
                "bytes": item.get("bytes"),
            }
        )
    return content, normalized if normalized else None


def _extract_generate_payload(data: dict[str, Any]) -> tuple[str, list | None]:
    """
    Parse native /api/generate response:
      response: text
      logprobs: [{token, logprob, top_logprobs, ...}, ...]
    """
    text = (data.get("response") or "").strip() if isinstance(data, dict) else ""
    lp = data.get("logprobs") if isinstance(data, dict) else None
    return text, lp if isinstance(lp, list) else None


def _merge_sampling_options(body: dict[str, Any], temperature: float | None) -> None:
    """Mutate Ollama request body to include sampling options (e.g. temperature for diverse semantic samples)."""
    if temperature is None:
        return
    opts = body.get("options")
    if not isinstance(opts, dict):
        opts = {}
    else:
        opts = dict(opts)
    opts["temperature"] = float(temperature)
    body["options"] = opts


def _generate_with_logprobs(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base: str,
    tpl: int,
    temperature: float | None,
) -> tuple[str, float | None, list | None, dict]:
    """
    Native generate endpoint with logprobs.
    Per article/reference this is the canonical endpoint for token-prob analysis.
    """
    url = f"{base}/api/generate"
    body: dict[str, Any] = {
        "model": model,
        "prompt": str(user_prompt),
        "stream": False,
        "logprobs": True,
        "top_logprobs": max(0, tpl),
    }
    if system_prompt and str(system_prompt).strip():
        body["system"] = str(system_prompt)
    _merge_sampling_options(body, temperature)
    data = _post_json(url, body)
    text, raw_lp = _extract_generate_payload(data)
    return text, sum_chosen_token_logprobs(raw_lp), raw_lp, data


def _native_chat_with_logprobs(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base: str,
    tpl: int,
    temperature: float | None,
) -> tuple[str, float | None, list | None, dict]:
    url = f"{base}/api/chat"
    messages: list[dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(user_prompt)})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "logprobs": True,
        "top_logprobs": max(0, tpl),
    }
    _merge_sampling_options(body, temperature)
    data = _post_json(url, body)
    text, raw_lp = _extract_native_chat_payload(data)
    return text, sum_chosen_token_logprobs(raw_lp), raw_lp, data


def _openai_compat_chat_with_logprobs(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base: str,
    tpl: int,
    temperature: float | None,
) -> tuple[str, float | None, list | None, dict]:
    url = f"{base}/v1/chat/completions"
    messages: list[dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(user_prompt)})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "logprobs": True,
        "top_logprobs": max(0, tpl),
    }
    _merge_sampling_options(body, temperature)
    data = _post_json(url, body)
    text, raw_lp = _extract_openai_chat_payload(data)
    return text, sum_chosen_token_logprobs(raw_lp), raw_lp, data


def ollama_chat_with_logprobs(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    host: str | None = None,
    top_logprobs: int | None = None,
    temperature: float | None = None,
) -> tuple[str, float | None, list | None, dict]:
    """
    Try multiple Ollama endpoints with logprobs enabled.

    Returns:
        (assistant_text, sequence_logprob_sum, raw_token_logprobs_list, raw_json)
    """
    if requests is None:
        raise RuntimeError("requests is required for Ollama logprob chat")

    base = (host or OLLAMA_HOST).rstrip("/")
    tpl = top_logprobs if top_logprobs is not None else TOP_LOGPROBS

    errors: list[str] = []
    for name, fn in (
        ("native_api_generate", _generate_with_logprobs),
        ("native_api_chat", _native_chat_with_logprobs),
        ("openai_compat_chat", _openai_compat_chat_with_logprobs),
    ):
        try:
            text, seq, raw_lp, data = fn(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base=base,
                tpl=tpl,
                temperature=temperature,
            )
            if raw_lp and seq is not None:
                return text, seq, raw_lp, data
            # Endpoint succeeded but gave no token logprobs; keep trying fallbacks.
            errors.append(f"{name}: succeeded_without_logprobs")
        except Exception as e:
            errors.append(f"{name}: {e}")

    # No endpoint yielded logprobs; raise a clear error for caller fallback handling.
    raise RuntimeError(
        "Ollama logprobs unavailable on tested endpoints. "
        + "; ".join(errors)
    )
