#!/usr/bin/env python3
"""
Diagnose Ollama logprob support across installed local models.

By default, this targets "other" models (excluding deepseek/llama bases) so you
can quickly verify whether models like Qwen-based agent models return logprobs.

Examples:
  # Default: all installed models except llama/deepseek bases
  python scripts/test_ollama_logprobs_diagnostic.py

  # Test specific models only
  python scripts/test_ollama_logprobs_diagnostic.py --models MathExpert:latest mistral:latest

  # Include llama/deepseek too
  python scripts/test_ollama_logprobs_diagnostic.py --include-baselines
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure `src.*` imports work when this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASELINE_STEMS = frozenset({"deepseek-r1", "llama3.2", "llama3"})
BASELINE_MODELS = ("deepseek-r1:latest", "llama3.2:latest", "llama3:latest")


def _stem(model_name: str) -> str:
    return model_name.split(":", 1)[0].strip().lower()


def list_installed_ollama_models() -> list[str]:
    proc = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    out: list[str] = []
    for ln in lines[1:]:
        parts = ln.split()
        if parts:
            out.append(parts[0])
    return out


def run_probe(model: str, prompt: str, system_prompt: str) -> dict:
    from src.manager.ollama_logprobs import ollama_chat_with_logprobs
    import ollama

    t0 = time.time()
    try:
        text, seq_lp, raw_lp, raw = ollama_chat_with_logprobs(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        elapsed = time.time() - t0
        has_token_logprobs = isinstance(raw_lp, list) and len(raw_lp) > 0
        has_sequence_logprob = seq_lp is not None
        raw_has_logprobs = bool(isinstance(raw, dict) and raw.get("logprobs") is not None)
        raw_msg_has_logprobs = bool(
            isinstance(raw, dict)
            and isinstance(raw.get("message"), dict)
            and raw["message"].get("logprobs") is not None
        )
        return {
            "model": model,
            "ok": True,
            "elapsed_s": round(elapsed, 3),
            "has_token_logprobs": has_token_logprobs,
            "raw_has_logprobs": raw_has_logprobs,
            "raw_message_has_logprobs": raw_msg_has_logprobs,
            "token_logprob_steps": len(raw_lp) if isinstance(raw_lp, list) else 0,
            "has_sequence_logprob": has_sequence_logprob,
            "sequence_logprob": seq_lp,
            "response_preview": (text or "")[:140],
        }
    except Exception as e:
        # Distinguish "no logprobs support" from general request failures by
        # checking whether regular chat still succeeds.
        try:
            plain = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            plain_text = (
                plain.get("message", {}).get("content", "")
                if isinstance(plain, dict)
                else ""
            )
            elapsed = time.time() - t0
            return {
                "model": model,
                "ok": True,
                "elapsed_s": round(elapsed, 3),
                "has_token_logprobs": False,
                "raw_has_logprobs": False,
                "raw_message_has_logprobs": False,
                "token_logprob_steps": 0,
                "has_sequence_logprob": False,
                "sequence_logprob": None,
                "error": str(e),
                "fallback_plain_chat_ok": True,
                "response_preview": (plain_text or "")[:140],
            }
        except Exception:
            pass
        elapsed = time.time() - t0
        return {
            "model": model,
            "ok": False,
            "elapsed_s": round(elapsed, 3),
            "has_token_logprobs": False,
            "token_logprob_steps": 0,
            "has_sequence_logprob": False,
            "sequence_logprob": None,
            "error": str(e),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check which Ollama models return usable logprobs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model names to test (e.g., MathExpert:latest mistral:latest).",
    )
    parser.add_argument(
        "--append-baselines",
        action="store_true",
        help=(
            "Append baseline models to explicit --models: "
            "deepseek-r1:latest llama3.2:latest llama3:latest."
        ),
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include deepseek/llama baselines in auto-discovery mode.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In one sentence, explain why regular testing helps software quality.",
        help="User prompt for the probe.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a concise assistant.",
        help="System prompt for the probe.",
    )
    parser.add_argument(
        "--save-jsonl",
        action="store_true",
        help="Write results to scripts/logs/ollama_logprobs_diag_<timestamp>.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from src.manager.ollama_logprobs import (
        get_ollama_server_version,
        ollama_logprobs_likely_supported,
    )

    ver = get_ollama_server_version()
    support_hint = ollama_logprobs_likely_supported()
    print(f"[diag] Ollama server version: {ver or 'unknown'}")
    if support_hint is False:
        print(
            "[diag] WARNING: This Ollama version is likely too old for chat/generate logprobs "
            "(article suggests >= 0.12.11). Consider upgrading/restarting Ollama."
        )
    elif support_hint is None:
        print("[diag] Could not determine if this Ollama version supports logprobs.")
    else:
        print("[diag] Version appears new enough for logprobs; probing endpoints/models next.")

    try:
        if args.models:
            targets = list(args.models)
            if args.append_baselines:
                targets.extend(BASELINE_MODELS)
            targets = list(dict.fromkeys(targets))
        else:
            installed = list_installed_ollama_models()
            targets = [
                m
                for m in installed
                if args.include_baselines or _stem(m) not in BASELINE_STEMS
            ]
    except Exception as e:
        print(f"[diag] Failed to list installed Ollama models: {e}", file=sys.stderr)
        return 1

    if not targets:
        print("[diag] No target models found to test.")
        return 0

    print(f"[diag] Testing {len(targets)} model(s): {targets}")
    results: list[dict] = []
    for model in targets:
        print(f"\n[diag] Probing {model} ...")
        r = run_probe(model=model, prompt=args.prompt, system_prompt=args.system_prompt)
        results.append(r)
        if r["ok"]:
            print(
                f"  ok=True token_logprobs={r['has_token_logprobs']} "
                f"raw_logprobs={r.get('raw_has_logprobs') or r.get('raw_message_has_logprobs')} "
                f"steps={r['token_logprob_steps']} seq_lp={r['has_sequence_logprob']} "
                f"elapsed={r['elapsed_s']}s"
            )
        else:
            print(f"  ok=False error={r.get('error')} elapsed={r['elapsed_s']}s")

    print("\n===== OLLAMA LOGPROB DIAGNOSTIC SUMMARY =====")
    ok = [r for r in results if r.get("ok")]
    with_lp = [r for r in ok if r.get("has_token_logprobs")]
    no_lp = [r for r in ok if not r.get("has_token_logprobs")]
    failed = [r for r in results if not r.get("ok")]
    print(f"- total tested: {len(results)}")
    print(f"- request succeeded: {len(ok)}")
    print(f"- succeeded with token logprobs: {len(with_lp)}")
    print(f"- succeeded without token logprobs: {len(no_lp)}")
    print(f"- request failed: {len(failed)}")

    if with_lp:
        print("\nModels WITH token logprobs:")
        for r in with_lp:
            print(f"  - {r['model']} (steps={r['token_logprob_steps']})")

    if no_lp:
        print("\nModels WITHOUT token logprobs (chat worked):")
        for r in no_lp:
            print(f"  - {r['model']}")

    if failed:
        print("\nModels FAILED (request error):")
        for r in failed:
            print(f"  - {r['model']}: {r.get('error')}")

    if args.save_jsonl:
        out_dir = Path(__file__).resolve().parent / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ollama_logprobs_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n[diag] Saved JSONL results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

