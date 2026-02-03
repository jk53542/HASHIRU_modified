# src/metrics/semantic_metrics.py
"""
Wrapper module exposing compute_semantic_entropy(...) and compute_semantic_density(...)
to the rest of HASHIRU. Currently this file uses dummy values instead of the core functions:
   - semantic_entropy_impl(prompt, response, samples=..., **kwargs)
   - semantic_density_impl(prompt, response, samples=..., **kwargs)

If your implementations live elsewhere, change the imports accordingly.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Import the user's implementations (update import path if different)
# from your_impl_module import semantic_entropy_impl, semantic_density_impl

def compute_semantic_entropy(prompt: str, response: str, samples: list = None, **kwargs) -> dict:
    """
    Returns a dict:
    {
      "entropy": float,             # main scalar
      "clusters": [...],            # optional cluster info for debugging
      "sample_count": int,
      "diagnostics": {...}          # optional debug info (timings etc.)
    }
    """
    t0 = time.time()

    # result = semantic_entropy_impl(prompt=prompt, response=response, samples=samples, **kwargs)
    # let's test some values, for now with some hardcoding
    result = 1

    dt = time.time() - t0
    # out = {
    #     "entropy": float(result.get("entropy", result)),  # accept both styles
    #     "clusters": result.get("clusters", None),
    #     "sample_count": len(samples) if samples is not None else result.get("sample_count", None),
    #     "diagnostics": {"elapsed_s": dt}
    # }
    # logger.debug("Computed semantic_entropy: %s", out)

    # return out
    return {
        "entropy": result,
        "clusters": None,
        "sample_count": len(samples) if samples else None,
        "diagnostics": {"elapsed_s": dt}
    }

def compute_semantic_density(prompt: str, response: str, samples: list = None, **kwargs) -> dict:
    """
    Returns a dict:
    {
      "density": float,             # main scalar in [0,1], higher = more confident
      "kernel_values": [...],       # optional
      "sample_count": int,
      "diagnostics": {...}
    }
    """
    t0 = time.time()

    # result = semantic_density_impl(prompt=prompt, response=response, samples=samples, **kwargs)
    # let's test some values, for now with some hardcoding
    result = 0.1


    dt = time.time() - t0
    # out = {
    #     "density": float(result.get("density", result)),
    #     "kernel_values": result.get("kernel_values", None),
    #     "sample_count": len(samples) if samples is not None else result.get("sample_count", None),
    #     "diagnostics": {"elapsed_s": dt}
    # }
    # logger.debug("Computed semantic_density: %s", out)

    return {
        "density": result,
        "kernel_values": None,
        "sample_count": len(samples) if samples else None,
        "diagnostics": {"elapsed_s": dt}
    }
    # return out
