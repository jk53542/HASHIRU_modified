"""
Standalone test script for semantic entropy and semantic density calculations.

Uses the same pipeline as semantic_metrics_tool.py (gateway at HASHIRU_METRICS_SERVICE_URL)
so you can verify the metrics without HASHIRU. Requires the metrics gateway and backends
(entropy + density services) to be running.

Expected behavior:
  - Semantic entropy: 0 = all responses in one semantic cluster; higher = more distinct meanings.
  - Semantic density: 1 = all responses very similar (high confidence); 0 = very diverse (low).

Run from project root (HASHIRU_modified) so 'src' is on the path:
  cd HASHIRU_modified && python -m src.test_semantic_entropy_density

Or: PYTHONPATH=. python src/test_semantic_entropy_density.py
"""

from __future__ import annotations

import logging
import os
import sys

# Verbose logging so you can see every example and metric
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Reduce noise from requests/urllib3 if desired
logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_test(
    name: str,
    prompt: str,
    response: str,
    samples: list[str] | None,
    expected_entropy_note: str,
    expected_density_note: str,
) -> None:
    """Call compute_semantic_metrics_both and log inputs + outputs."""
    from src.metrics.semantic_metrics import compute_semantic_metrics_both

    log.info("")
    log.info("=" * 80)
    log.info("TEST: %s", name)
    log.info("=" * 80)
    log.info("Expected (for reference): entropy %s; density %s", expected_entropy_note, expected_density_note)
    log.info("")
    log.info("PROMPT (%d chars):", len(prompt))
    log.info("  %s", prompt[:500] + "..." if len(prompt) > 500 else prompt)
    log.info("")
    log.info("PRIMARY RESPONSE (%d chars):", len(response))
    log.info("  %s", response[:400] + "..." if len(response) > 400 else response)
    log.info("")
    if samples:
        log.info("SAMPLES (%d):", len(samples))
        for i, s in enumerate(samples):
            log.info("  [%d] (%d chars) %s", i + 1, len(s), (s[:300] + "..." if len(s) > 300 else s))
    else:
        log.info("SAMPLES: None (entropy will be 0, density may be 0.5 or 1.0)")
    log.info("")

    both = compute_semantic_metrics_both(prompt=prompt, response=response, samples=samples)

    entropy = both.get("entropy", 0.0)
    density = both.get("density", 0.5)
    diag = both.get("diagnostics") or {}

    log.info("RESULTS:")
    log.info("  semantic_entropy = %s", entropy)
    log.info("  semantic_density = %s", density)
    log.info("  diagnostics: elapsed_s=%s, entropy_ok=%s, density_ok=%s",
             diag.get("elapsed_s"), diag.get("entropy_ok"), diag.get("density_ok"))
    if diag.get("entropy_error"):
        log.info("  entropy_error: %s", diag["entropy_error"])
    if diag.get("density_error"):
        log.info("  density_error: %s", diag["density_error"])
    if both.get("clusters") is not None:
        log.info("  clusters (entropy): %s", both["clusters"])
    if both.get("kernel_values") is not None:
        log.info("  kernel_values (density): %s", both["kernel_values"])
    log.info("")


def main() -> None:
    # Extend timeout for tests (gateway + entropy/density backends can be slow, especially first load)
    os.environ.setdefault("HASHIRU_METRICS_TIMEOUT", "120")
    timeout_sec = int(os.getenv("HASHIRU_METRICS_TIMEOUT", "30"))
    log.info("Metrics request timeout: %s seconds (set HASHIRU_METRICS_TIMEOUT to override)", timeout_sec)

    gateway = os.getenv("HASHIRU_METRICS_SERVICE_URL", "http://127.0.0.1:8123")
    log.info("Using metrics gateway: %s (set HASHIRU_METRICS_SERVICE_URL to override)", gateway)
    log.info("")

    # -------------------------------------------------------------------------
    # Case 1: IDENTICAL RESPONSES (extreme: same string repeated)
    # Expected: LOW entropy (one cluster), HIGH density (all identical)
    # -------------------------------------------------------------------------
    prompt_weather = "What is the weather like today?"
    response_weather = "It is sunny and warm with a high of 75 degrees."
    samples_identical = [
        "It is sunny and warm with a high of 75 degrees.",
        "It is sunny and warm with a high of 75 degrees.",
        "It is sunny and warm with a high of 75 degrees.",
    ]
    run_test(
        name="Identical responses (same string x4)",
        prompt=prompt_weather,
        response=response_weather,
        samples=samples_identical,
        expected_entropy_note="LOW (0 or near 0) — all in one cluster",
        expected_density_note="HIGH (near 1) — all identical",
    )

    # -------------------------------------------------------------------------
    # Case 2: PARAPHRASES (same meaning, different wording)
    # Expected: LOW entropy (one cluster), HIGH density (similar)
    # -------------------------------------------------------------------------
    samples_paraphrase = [
        "The weather is nice: sunny and around 75°F.",
        "Sunny and warm today, high about 75.",
        "It's a beautiful day—sunny, warm, 75 degrees.",
    ]
    run_test(
        name="Paraphrases (same meaning)",
        prompt=prompt_weather,
        response=response_weather,
        samples=samples_paraphrase,
        expected_entropy_note="LOW — same semantic cluster",
        expected_density_note="HIGH — high pairwise similarity",
    )

    # -------------------------------------------------------------------------
    # Case 3: COMPLETELY DIFFERENT TOPICS (unrelated answers to same prompt)
    # Expected: HIGH entropy (multiple clusters), LOW density (low similarity)
    # -------------------------------------------------------------------------
    samples_different_topics = [
        "The capital of France is Paris.",
        "I prefer chocolate ice cream over vanilla.",
        "The mitochondria are the powerhouse of the cell.",
        "Python is a programming language.",
        "The meeting is scheduled for 3 PM tomorrow.",
    ]
    run_test(
        name="Completely different topics (unrelated answers)",
        prompt=prompt_weather,
        response=response_weather,
        samples=samples_different_topics,
        expected_entropy_note="HIGH — many semantic clusters",
        expected_density_note="LOW (near 0) — low pairwise similarity",
    )

    # -------------------------------------------------------------------------
    # Case 4: MIXED (some similar, some off-topic)
    # Expected: medium entropy, medium density
    # -------------------------------------------------------------------------
    samples_mixed = [
        "It is sunny and warm with a high of 75 degrees.",  # same as response
        "Sunny and warm, about 75°F.",                     # paraphrase
        "I like pizza.",                                    # off-topic
        "The stock market went up today.",                  # off-topic
    ]
    run_test(
        name="Mixed (some similar, some off-topic)",
        prompt=prompt_weather,
        response=response_weather,
        samples=samples_mixed,
        expected_entropy_note="MEDIUM — 2+ clusters",
        expected_density_note="MEDIUM — mixed similarity",
    )

    # -------------------------------------------------------------------------
    # Case 5: NO SAMPLES (single response only)
    # Expected: entropy = 0 (no clustering), density often 0.5 or 1.0 (backend-dependent)
    # -------------------------------------------------------------------------
    run_test(
        name="No samples (single response)",
        prompt=prompt_weather,
        response=response_weather,
        samples=None,
        expected_entropy_note="0 (only one response)",
        expected_density_note="0.5 or 1.0 (no pairwise comparison)",
    )

    # -------------------------------------------------------------------------
    # Case 6: CONTRADICTORY (same topic, opposite meaning)
    # Expected: entropy can be higher (different clusters), density lower than paraphrases
    # -------------------------------------------------------------------------
    samples_contradictory = [
        "It is rainy and cold with a high of 45 degrees.",
        "The weather is terrible—overcast and chilly.",
        "It's snowing and below freezing.",
    ]
    run_test(
        name="Contradictory (opposite meaning, same topic)",
        prompt=prompt_weather,
        response=response_weather,
        samples=samples_contradictory,
        expected_entropy_note="HIGHER than paraphrases — different meaning clusters",
        expected_density_note="LOWER than paraphrases — less similar",
    )

    log.info("=" * 80)
    log.info("ALL TESTS COMPLETE")
    log.info("=" * 80)
    log.info("Check that: (1) Identical/paraphrase cases give LOW entropy, HIGH density.")
    log.info("            (2) Different-topic cases give HIGH entropy, LOW density.")
    log.info("If the gateway or backends are down, entropy/density may be 0.0 and 0.5.")


if __name__ == "__main__":
    main()
