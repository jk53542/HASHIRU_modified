# src/metrics/semantic_metrics.py
"""
Wrapper module exposing compute_semantic_entropy(...) and compute_semantic_density(...)
to the rest of HASHIRU. Calls out to separate services (semantic_uncertainty and
semantic-density) running in their own environments to avoid dependency conflicts.
See INTEGRATION_SEMANTIC_METRICS.md for required inputs and how to run the services.

**Sequence log-likelihoods (open-weight via Ollama):**
For Ollama-backed agents whose **base model** is an open-weight family we support
(e.g. DeepSeek-R1, Llama 3.2), HASHIRU can attach **summed chosen-token log-probs**
(one float per completion) when `HASHIRU_OLLAMA_LOGPROBS` is enabled and the local
Ollama build exposes chat `logprobs`. Those values are sent to the metrics gateway
inside **`metadata["sequence_logprobs"]`**, aligned with `[primary_response] + samples`
so backends may use them without replacing the multi-sample pipeline.

**Other providers:** Cloud APIs may not return comparable logprobs; the text-sample
path remains the default for entropy/density.
"""

import os
import time
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

# Base URL for the metrics service (gateway). Override via env if needed.
METRICS_SERVICE_URL = os.getenv("HASHIRU_METRICS_SERVICE_URL", "http://127.0.0.1:8123")
# Timeout: first request may need to wait for DeBERTa load on entropy backend; use 120s default
METRICS_TIMEOUT = int(os.getenv("HASHIRU_METRICS_TIMEOUT", "120"))


def _maybe_trim_for_metrics(text: str) -> str:
    """
    If HASHIRU_SEMANTIC_METRICS_TEXT_TAIL_CHARS is a positive integer, keep only the last N
    characters of each string sent to entropy/density. Useful when long shared reasoning
    prefixes swamp embedding/entailment signal but the conclusion differs at the end.
    """
    raw = os.getenv("HASHIRU_SEMANTIC_METRICS_TEXT_TAIL_CHARS", "").strip()
    if not raw:
        return text
    try:
        n = int(raw)
    except ValueError:
        return text
    if n <= 0 or not isinstance(text, str):
        return text
    t = text.strip()
    return t[-n:] if len(t) > n else t


def _build_payload(prompt: str, response: str, samples: list = None, **kwargs) -> dict:
    """Build JSON payload for the metrics service. responses = [response]; samples = [samples] per response."""
    response = _maybe_trim_for_metrics(response or "")
    # Distinguish None (omit samples) from [] (explicit empty list from caller).
    if samples is None:
        trimmed_samples = None
    else:
        trimmed_samples = [_maybe_trim_for_metrics(s) for s in samples]
    responses = [response]
    # Service expects samples[i] to align with responses[i]. One response -> one list of samples.
    sample_lists = [list(trimmed_samples)] if trimmed_samples is not None else None
    metadata: dict = {}
    md = kwargs.get("metadata")
    if isinstance(md, dict):
        metadata = dict(md)
    seq = kwargs.get("sequence_logprobs")
    if seq is not None:
        metadata["sequence_logprobs"] = seq
    return {
        "prompt": prompt,
        "responses": responses,
        "samples": sample_lists,
        "metadata": metadata,
    }


def compute_semantic_metrics_both(prompt: str, response: str, samples: list = None, **kwargs) -> dict:
    """
    Single gateway call returning both entropy and density (avoids two round-trips and
    ensures both values come from the same backend response; first load of DeBERTa can be slow).
    Returns dict with keys: entropy, density, diagnostics, clusters, kernel_values.
    """
    t0 = time.time()
    out = {
        "entropy": 0.0,
        "density": 0.5,
        "clusters": None,
        "kernel_values": None,
        "sample_count": len(samples) if samples else None,
        "diagnostics": {"elapsed_s": 0.0, "entropy_ok": False, "density_ok": False},
    }
    if requests is None:
        logger.warning("requests not installed; semantic metrics fallback")
        out["diagnostics"]["elapsed_s"] = time.time() - t0
        return out
    url = f"{METRICS_SERVICE_URL.rstrip('/')}/score"
    payload = _build_payload(prompt, response, samples, **kwargs)
    try:
        r = requests.post(url, json=payload, timeout=METRICS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        reasons = data.get("reasons") or {}
        entropies = data.get("entropy", [])
        densities = data.get("density", [])
        logger.info("gateway response: entropy=%s, density=%s, reasons_keys=%s", entropies, densities, list(reasons.keys()))
        out["diagnostics"]["entropy_ok"] = reasons.get("entropy_ok", True)
        out["diagnostics"]["density_ok"] = reasons.get("density_ok", True)
        out["diagnostics"]["used_samples"] = reasons.get("used_samples", bool(samples))
        if reasons.get("entropy_error"):
            out["diagnostics"]["entropy_error"] = reasons["entropy_error"]
            logger.warning("Entropy backend: %s", reasons["entropy_error"])
        if reasons.get("density_error"):
            out["diagnostics"]["density_error"] = reasons["density_error"]
            logger.warning("Density backend: %s", reasons["density_error"])
        raw_e = float(entropies[0]) if entropies else 0.0
        raw_d = float(densities[0]) if densities else 0.5
        out["entropy"] = max(0.0, raw_e)
        out["density"] = max(0.0, min(1.0, raw_d))
        out["clusters"] = reasons.get("clusters")
        out["kernel_values"] = reasons.get("kernel_values")
        out["sample_count"] = len(samples) if samples else reasons.get("n")
        if reasons.get("entropy_modes") is not None:
            out["diagnostics"]["entropy_modes"] = reasons["entropy_modes"]
        if reasons.get("used_sequence_logprobs") is not None:
            out["diagnostics"]["used_sequence_logprobs"] = reasons["used_sequence_logprobs"]
    except Exception as e:
        logger.warning("Metrics gateway request failed: %s", e)
        out["diagnostics"]["error"] = str(e)
    out["diagnostics"]["elapsed_s"] = time.time() - t0
    return out


def compute_semantic_entropy(prompt: str, response: str, samples: list = None, **kwargs) -> dict:
    """
    Returns a dict:
    {
      "entropy": float,             # main scalar
      "clusters": [...],            # optional cluster info for debugging
      "sample_count": int,
      "diagnostics": {...}          # optional debug info (timings etc.)
    }
    Calls the metrics service (semantic_uncertainty) when available; otherwise returns a fallback.
    """
    t0 = time.time()
    out = {
        "entropy": 0.0,
        "clusters": None,
        "sample_count": len(samples) if samples else None,
        "diagnostics": {"elapsed_s": 0.0},
    }

    if requests is None:
        logger.warning("requests not installed; semantic_entropy fallback")
        out["diagnostics"]["elapsed_s"] = time.time() - t0
        return out

    url = f"{METRICS_SERVICE_URL.rstrip('/')}/score"
    payload = _build_payload(prompt, response, samples, **kwargs)

    try:
        r = requests.post(url, json=payload, timeout=METRICS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        reasons = data.get("reasons") or {}
        out["diagnostics"]["entropy_ok"] = reasons.get("entropy_ok", True)
        out["diagnostics"]["used_samples"] = reasons.get("used_samples", bool(samples))
        if reasons.get("entropy_error"):
            out["diagnostics"]["entropy_error"] = reasons["entropy_error"]
            logger.warning("Entropy backend failed: %s", reasons["entropy_error"])
        # Service returns ScoreResponse: entropy=list, density=list (one per response)
        entropies = data.get("entropy", [])
        raw_e = float(entropies[0]) if entropies else 0.0
        out["entropy"] = max(0.0, raw_e)  # entropy is non-negative
        out["clusters"] = reasons.get("clusters")
        out["sample_count"] = len(samples) if samples else reasons.get("n")
        if not samples and out["entropy"] == 0.0:
            logger.info("Semantic entropy is 0.0 because only one response was sent; pass multiple samples for meaningful entropy.")
    except Exception as e:
        logger.warning("Metrics service request failed for semantic_entropy: %s", e)
        out["entropy"] = 0.0
        out["diagnostics"]["error"] = str(e)

    out["diagnostics"]["elapsed_s"] = time.time() - t0
    return out

def compute_semantic_density(prompt: str, response: str, samples: list = None, **kwargs) -> dict:
    """
    Returns a dict:
    {
      "density": float,             # main scalar in [0,1], higher = more confident
      "kernel_values": [...],       # optional
      "sample_count": int,
      "diagnostics": {...}
    }
    Calls the metrics service (semantic-density) when available; otherwise returns a fallback.
    """
    t0 = time.time()
    out = {
        "density": 0.5,
        "kernel_values": None,
        "sample_count": len(samples) if samples else None,
        "diagnostics": {"elapsed_s": 0.0},
    }

    if requests is None:
        logger.warning("requests not installed; semantic_density fallback")
        out["diagnostics"]["elapsed_s"] = time.time() - t0
        return out

    url = f"{METRICS_SERVICE_URL.rstrip('/')}/score"
    payload = _build_payload(prompt, response, samples, **kwargs)

    try:
        r = requests.post(url, json=payload, timeout=METRICS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        reasons = data.get("reasons") or {}
        out["diagnostics"]["density_ok"] = reasons.get("density_ok", True)
        out["diagnostics"]["used_samples"] = reasons.get("used_samples", bool(samples))
        if reasons.get("density_error"):
            out["diagnostics"]["density_error"] = reasons["density_error"]
            logger.warning("Density backend failed: %s", reasons["density_error"])
        densities = data.get("density", [])
        raw_d = float(densities[0]) if densities else 0.5
        # Density is confidence in [0, 1]; clamp in case backend returns raw similarity
        out["density"] = max(0.0, min(1.0, raw_d))
        out["kernel_values"] = reasons.get("kernel_values")
        out["sample_count"] = len(samples) if samples else reasons.get("n")
        if not samples and out["density"] in (0.5, 1.0):
            logger.info("Semantic density with one response only; pass multiple samples for meaningful density.")
    except Exception as e:
        logger.warning("Metrics service request failed for semantic_density: %s", e)
        out["density"] = 0.5
        out["diagnostics"]["error"] = str(e)

    out["diagnostics"]["elapsed_s"] = time.time() - t0
    return out
