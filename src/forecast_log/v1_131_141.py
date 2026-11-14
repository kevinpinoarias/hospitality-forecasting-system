"""
Versions v1.131 and v1.141 of the models behind the forecast log: v1.13 and v1.14 served with the hybrid stand-in for the manager's number.

What they are (asked for by Kevin on 14 November 2026, after Series 88 and 89 in docs/EXPERIMENT_LOG.md):
  * v1.131 = v1.13 and v1.141 = v1.14, with NO refit: every fitted piece is a byte-for-byte copy of the parent's own files (the fingerprints are checked against the parent's registration),
    the same training rows and cut-off, the same correction history, the same weekend options.
  * the only difference is the number fed to the models in the place of the manager's number, which is not known in time: the HYBRID (S6) in place of the weekday-and-season median (S0) every
    version up to v1.14 uses. S6 is the same date a year earlier from 24 December to 2 January and the closed-day-aware median on every other day (src/forecast_log/standin.py).
  * so each differs from its parent on every day the two stand-ins differ (most of the year the gap is small; around Christmas and New Year it is large), and Q9 and Q10 answer one question each:
    Q9 v1.131 against v1.13, Q10 v1.141 against v1.14.
  * registered in `models/versions.json` with `"standin": "S6"`; the rows of the hydro arm have model `hydro_v1.131` / `hydro_v1.141`, and `manager_forecast_used` is the stand-in the version was given.

Real registration (clean checkout; copies the parent's folder into models/<version>):
  python -m src.forecast_log.v1_131_141 register --version v1.131 --log-repo D:\\Projectos\\DRG\\forecasting-log
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.experiments import run_series35_holdout as h
from src.forecast_log import standin, v1

PARENTS = {"v1.131": "v1.13", "v1.141": "v1.14"}


def arm_name(version: str) -> str:
    return f"hydro_{version}"


def register(log_repo: Path, version: str) -> dict:
    """Copy the parent's fitted pieces into models/<version> and register the version (append-only). Refuses if the version is unknown or already registered, if the parent is not registered, if the
    folder is not empty, if the correction history is missing or differs from the parent's, if a copied piece does not match the fingerprint its parent registered, or if the checkout is dirty."""
    if version not in PARENTS:
        raise RuntimeError(f"{version} is not one of {sorted(PARENTS)}")
    parent_name = PARENTS[version]
    versions_path = log_repo / "models" / "versions.json"
    registry = json.loads(versions_path.read_text(encoding="utf-8"))
    if any(v["version"] == version for v in registry["versions"]):
        raise RuntimeError(f"version {version} is already registered in {versions_path}; a change must be a new version")
    parent = next((v for v in registry["versions"] if v["version"] == parent_name), None)
    if parent is None:
        raise RuntimeError(f"the parent version {parent_name} is not registered in {versions_path}")
    models_root = v1.MODELS_DIR.parent
    target = models_root / version
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"{target} is not empty; models are copied once and never overwritten")
    history = v1.correction_history_path(version)
    if not history.exists():
        raise RuntimeError(f"{history} is missing: {version} has {parent_name}'s cut-off, so it uses {parent_name}'s correction history; copy data/training/{parent_name}_correction_history.csv to that name")
    if h.file_sha256(history) != parent["correction_history_sha256"]:
        raise RuntimeError(f"{history} is not the correction history {parent_name} recorded")
    commit = v1.code_commit()
    pieces = {}
    for name, spec in parent["pieces"].items():
        shutil.copytree(models_root / parent_name / name, target / name)
        got = h.fingerprint(target / name)
        if got != spec["fingerprint"]:
            raise RuntimeError(f"the copied piece {name} does not match the fingerprint {parent_name} registered; {target} must be removed before another attempt")
        pieces[name] = {**spec, "how": f"copied from {parent_name}"}
    entry = {
        **{k: v for k, v in parent.items() if k not in ("version", "parent", "fitted_at", "code_commit", "arms", "change_from_parent", "pieces", "standin", "note")},
        "version": version, "parent": parent_name, "fitted_at": datetime.now().astimezone().isoformat(timespec="seconds"), "code_commit": commit,
        "arms": [arm_name(version)], "standin": standin.NAME,
        "change_from_parent": f"no refit: every piece is {parent_name}'s own file; the number fed to the models in place of the manager's number is the hybrid stand-in {standin.NAME} "
                              f"(src/forecast_log/standin.py: the manager's number on the same date a year earlier from 24 December to 2 January, the closed-day-aware weekday-and-season median on every other day) "
                              f"instead of the weekday-and-season median S0 that {parent_name} is served with",
        "pieces": pieces,
        "note": "CatBoost writes a random id into every saved model, so these fingerprints identify THESE files (copies of the parent's); a refit is a new version.",
    }
    registry["versions"].append(entry)
    versions_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"[{version}] registered in {versions_path}", flush=True)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    real = sub.add_parser("register", help="the registration (after the pre-registration amendment)")
    real.add_argument("--version", required=True, choices=sorted(PARENTS))
    real.add_argument("--log-repo", type=Path, required=True)
    args = parser.parse_args()
    register(args.log_repo, args.version)


if __name__ == "__main__":
    main()
