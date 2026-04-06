"""
Upper bound on how many times the CEO may delegate to workers (AskAgent / AskMultipleAgents)
within a single user message turn.

Each completed AskAgent or AskMultipleAgents tool call counts as one *worker tool round*.
The first round is the initial delegation; each additional round is a reprompt-style delegation.

Default: at most 5 reprompts after the first → 6 worker tool rounds total per user turn.

Override at runtime:
  export HASHIRU_MAX_CEO_WORKER_REPROMPTS=5
Set to 0 to allow only a single worker tool round (no reprompts).
"""

from __future__ import annotations

import os

# Module-level default if the environment variable is unset or invalid.
DEFAULT_MAX_CEO_WORKER_REPROMPTS = 5
_ENV_KEY = "HASHIRU_MAX_CEO_WORKER_REPROMPTS"


def max_ceo_worker_reprompts_per_user_turn() -> int:
    raw = os.getenv(_ENV_KEY, str(DEFAULT_MAX_CEO_WORKER_REPROMPTS)).strip()
    try:
        n = int(raw)
    except ValueError:
        n = DEFAULT_MAX_CEO_WORKER_REPROMPTS
    return max(0, min(n, 500))


def max_ceo_worker_tool_rounds_per_user_turn() -> int:
    """One initial delegation plus up to ``max_ceo_worker_reprompts_per_user_turn()`` reprompts."""
    return 1 + max_ceo_worker_reprompts_per_user_turn()
