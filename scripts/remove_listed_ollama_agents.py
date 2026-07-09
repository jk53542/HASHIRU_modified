#!/usr/bin/env python3
"""
Remove specific Ollama custom model names and drop matching keys from HASHIRU models.json.

Run from HASHIRU_modified (repo root containing src/models/models.json)::

    cd HASHIRU_modified
    python scripts/remove_listed_ollama_agents.py
    python scripts/remove_listed_ollama_agents.py --dry-run

Requires ``ollama`` on PATH. Restart the HASHIRU / Gradio server after running so it
reloads AgentManager state from disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Base names only (Ollama tags like :latest are stripped when running ollama rm).
AGENTS_TO_REMOVE: tuple[str, ...] = (
    "GeneralLawExpert",
    "MovieInformationExpert",
    "LegalAssistant",
    "MilitaryStrategyExpert",
    "NameAssociationExpert",
    "PetCareExpert",
    "SizeComparisonExpert",
    "ArtExpert",
    "NewsFactCheckerLlama",
    "FactCheckerAgent",
    "StoryTellerLlama",
    "ScamScriptWriter",
    "EntertainmentRecommender",
    "FinancialAnalyst",
    "NewsFactChecker",
    "AddictionExpert",
    "AnswerAgent",
    "PhilosophyExpert",
    "FareRulesAnalyst",
    "ReservationCancellationAgent",
    "AnimalRelationshipExpert",
    "StoryTellerSharp",
    "ReservationAssistant",
    "AnimalRaceAnalyst",
    "DemographicsAnalyst",
    "PhysicsAnalyst",
    "SuperstitionExplainer",
    "WorldAffairsAnalyst",
    "AstroAgent",
    "MobileDataTroubleshooter",
    "EconomicTrendsAnalyst",
    "CollegeAdmissionAdvisor",
    "MusicExpert",
    "WeatherComparer",
    "BiologyExpert",
    "CityDataAnalyst",
    "EthicalLegalAdvisor",
    "LogicExpert",
    "MedicalExpertAgent",
    "HistoricalInformationAgent",
    "LawExpert",
    "DiscountInvestigator",
    "SportsAnalyzer",
    "FranchiseComparer",
    "TravelAdvisor",
    "MovieReviewAgent",
    "MathExpert",
    "CookieComparer",
    "DreamAnalyzer",
    "PixarReviewer",
    "FoodAnalyzer",
    "StateComparerLite",
    "ReviewerGamma",
    "ReviewerAlpha",
    "ReviewerAlphaLite",
    "HypothermiaAnalyzer",
    "ToolCreatorAgent",
    "NeedleFearAnalyst",
    "NutritionAnalyst2",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_json_path() -> Path:
    return repo_root() / "src" / "models" / "models.json"


def ollama_rm(name: str, dry_run: bool) -> int:
    if dry_run:
        print(f"[dry-run] ollama rm {name}")
        return 0
    r = subprocess.run(
        ["ollama", "rm", name],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        print(f"ollama rm {name}: failed ({err or 'no message'})", file=sys.stderr)
    else:
        print(f"removed ollama model: {name}")
    return r.returncode


def prune_models_json(delete: set[str], dry_run: bool) -> None:
    path = models_json_path()
    if not path.exists():
        print(f"No models.json at {path}; skipping JSON prune", file=sys.stderr)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"{path} is not a JSON object; skipping", file=sys.stderr)
        return
    removed = [k for k in data if k in delete]
    kept = {k: v for k, v in data.items() if k not in delete}
    print(
        f"models.json: removing {len(removed)} keys ({', '.join(sorted(removed)) or 'none'}); "
        f"keeping {len(kept)}"
    )
    if dry_run:
        return
    path.write_text(json.dumps(kept, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not call ollama or write models.json",
    )
    args = p.parse_args()
    delete = set(AGENTS_TO_REMOVE)

    for name in sorted(delete):
        ollama_rm(name, args.dry_run)

    prune_models_json(delete, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
