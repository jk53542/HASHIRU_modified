"""
Open-weight base models HASHIRU may serve via Ollama.

- **DeepSeek-R1:** weights publicly released (e.g. MIT license on main artifacts).
- **Llama 3.2:** Meta Llama 3.2 Community License; weights publicly available.

Historical note:
This list was previously used to gate automatic logprob attachment. The current
pipeline now attempts logprobs for all Ollama models and falls back gracefully
when unavailable. This module remains useful for open-weight policy checks.
"""
from __future__ import annotations

# Must match AgentManager / models.json `base_model` for Ollama-backed agents.
OPEN_WEIGHT_OLLAMA_BASES = frozenset({"deepseek-r1", "llama3.2"})


def is_open_weight_ollama_base(base_model: str | None) -> bool:
    if not base_model:
        return False
    stem = base_model.split(":", 1)[0].strip().lower()
    return stem in OPEN_WEIGHT_OLLAMA_BASES
