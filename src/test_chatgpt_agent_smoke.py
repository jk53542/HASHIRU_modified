"""
Small smoke test for ChatGPT worker integration in HASHIRU.

What it validates:
1) Model routing: `chatgpt-5.4` resolves to AgentManager type `openai`.
2) Cost registry: `chatgpt-5.4` exists in AgentCostManager.
3) Agent creation path: when OPENAI_API_KEY is set, AgentManager can create
   and delete a ChatGPT-backed worker.

Run from project root:
  cd HASHIRU_modified && python -m src.test_chatgpt_agent_smoke
"""

from __future__ import annotations

import os
import sys
import time


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    from src.manager.agent_manager import AgentManager
    from src.tools.default_tools.agent_cost_manager import AgentCostManager

    print("[chatgpt-smoke] Starting ChatGPT worker smoke test...")

    # 1) Routing check
    am = AgentManager()
    routed = am._get_agent_type("chatgpt-5.4")
    _assert(routed == "openai", f"Expected agent type 'openai', got {routed!r}")
    print("[chatgpt-smoke] Routing OK: chatgpt-5.4 -> openai")

    # 2) Cost registry check
    costs = AgentCostManager().get_costs()
    _assert(
        "chatgpt-5.4" in costs,
        "chatgpt-5.4 is missing from AgentCostManager costs",
    )
    print("[chatgpt-smoke] Cost registry OK: chatgpt-5.4 present")

    # 3) Creation check (requires OPENAI_API_KEY)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "[chatgpt-smoke] SKIP creation: OPENAI_API_KEY is not set. "
            "Routing and registry checks passed."
        )
        return 0

    # Use a unique name to avoid collisions in models.json/persisted agents.
    agent_name = f"ChatGPTSmoke_{int(time.time())}"
    try:
        agent, _, _ = am.create_agent(
            agent_name=agent_name,
            base_model="chatgpt-5.4",
            system_prompt="You are a concise test assistant.",
            description="Smoke test agent for chatgpt-5.4 integration.",
            create_resource_cost=0,
            invoke_resource_cost=0,
            create_expense_cost=0,
            invoke_expense_cost=0,
            output_expense_cost=0,
        )
        _assert(agent.get_type() == "cloud", "Created agent type should be 'cloud'")
        print("[chatgpt-smoke] Creation OK: agent instantiated successfully")
    finally:
        try:
            am.delete_agent(agent_name)
            print("[chatgpt-smoke] Cleanup OK: smoke test agent deleted")
        except Exception:
            # If creation failed, cleanup may not be needed.
            pass

    print("[chatgpt-smoke] PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[chatgpt-smoke] FAIL: {e}", file=sys.stderr)
        raise

