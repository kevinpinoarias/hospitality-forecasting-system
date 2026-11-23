"""
Version v1.11 of the models behind the forecast log: v1 with the triangle payday intensity added to the two Saturday pieces.

STATUS (7 November 2026): Kevin decided the version's content and then that it goes into the log. The code is ready (this module, src/features/payday_intensity.py, the `also_run` setting of run_forecasts.py,
`predict.forecast_days(extras=...)`, Q4 in score_forecasts.py), OFF by default and NOT YET FITTED OR REGISTERED. Order of the real steps, each needing Kevin's go-ahead: (1) the amendment to the log's
PREREGISTRATION.md saying how v1.11 is scored (rule 1: how a later version is scored is written before its retrain), committed and pushed in the log repository; (2) commit and push this code, so the fit's recorded
code commit is a pushed one; (3) `fit` (below), which writes models/v1.11 and appends the entry to the log's models/versions.json; (4) copy models/v1.11, data/training/v1.11_correction_history.csv (a copy of
v1's) and the code to the runner and set `also_run: [v1.11]` in the runner's config/forecast_log.local.yaml. `fit-pieces` is a dry run into a folder given on the command line and registers nothing.

What it is (decided 7 November 2026, after Series 70-72 in docs/EXPERIMENT_LOG.md):
  * the hydro model plus `intensity_sym` (src/features/payday_intensity.py), in the Saturday pieces ONLY: `sat_A_lags` (Saturday with its two lags) and `raw` (the Saturday fallback). The Monday-to-Thursday,
    Friday and Sunday pieces get nothing new: the intensity's gain was only consistent on Saturday (Saturday -20.8 pounds a day, interval -35.9 to -6.2; every other weekday's interval includes zero and its sign flips between 2025 and 2026).
  * the same training data as v1: the rows before 2 November 2026 of the frozen features file, so the only difference from v1 is the new column.
  * the pieces that did not change are v1's own files, copied unchanged (same fingerprints), so v1 and v1.11 forecast identically on every day except Saturdays. The hydro arm needs five pieces:
    `log_hydro_thu`, `fri_A_all`, `sat_A_lags`, `raw` and `sun_B`. (The `log` piece belongs to the post arm only and is not part of v1.11.) The portfolio model is the frozen deployed model and is not part of it.
  * the new column is appended LAST to the two Saturday pieces' inputs (column order is part of a fit's definition).

Dry run (writes only to the folder given):
  python -m src.forecast_log.v1_11 fit-pieces --training-features <frozen v1_engineered_features.csv> --out <folder>
The real fit and registration (clean checkout; the frozen features file is in the runner folder, not the development folder):
  python -m src.forecast_log.v1_11 fit --log-repo <path-to>\\forecasting-log --training-features <frozen v1_engineered_features.csv>
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiments import run_series35_holdout as h
from src.features.payday_intensity import COLUMN as INTENSITY
from src.features.payday_intensity import add_payday_intensity
from src.forecast_log import v1
from src.models.final_model import DATE_COL, FINAL_FEATURES

VERSION = "v1.11"
PARENT = v1.VERSION
ARM = "hydro_v1.11"  # the model name of its rows in forecasts.csv (the log's de-duplication key has no version, so v1.11's rows need a name of their own)
SATURDAY_PIECES = ("sat_A_lags", "raw")
HYDRO_ARM_PIECES = ("log_hydro_thu", "fri_A_all", "sat_A_lags", "raw", "sun_B")
# name -> (extra inputs on top of the 38 final-model features, target); only the Saturday pieces differ from v1
PIECES: dict[str, tuple[list[str], str]] = {
    name: ((v1.PIECES[name][0] + [INTENSITY]) if name in SATURDAY_PIECES else v1.PIECES[name][0], v1.PIECES[name][1]) for name in HYDRO_ARM_PIECES
}


def training_frames(training_features: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The v1 training rows (plain, and with the weekend inputs added), each with the intensity column, from the FROZEN features file (its hash is checked as v1's is)."""
    sha = h.file_sha256(training_features)
    if sha != v1.TRAINING_FEATURES_SHA:
        raise RuntimeError(f"{training_features} is not the frozen features file (sha256 {sha[:16]}..., expected {v1.TRAINING_FEATURES_SHA[:16]}...)")
    full = v1.add_hydro_thu(h.load_engineered(training_features))
    train = h.complete_rows(full)
    train = train[train[DATE_COL] < v1.FIRST_EXCLUDED_DAY].reset_index(drop=True)
    train_w = h.with_weekend(train, full.set_index(DATE_COL)[v1.TARGET_COL])
    return add_payday_intensity(train, DATE_COL), add_payday_intensity(train_w, DATE_COL), sha


def fit_pieces(training_features: Path, out: Path, parent_models: Path | None = None) -> dict:
    """Dry run: fit the two Saturday pieces on the v1 training rows with the intensity added, and copy v1's other three hydro-arm pieces unchanged, into `out`. Nothing is registered."""
    parent_models = parent_models or v1.MODELS_DIR
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty; models are fitted once and never overwritten")
    train, train_w, sha = training_frames(training_features)
    pieces = {}
    for name, (extra, target) in PIECES.items():
        if name in SATURDAY_PIECES:
            frame = train_w if name in v1.WEEKEND_PIECES else train
            bag = h.Bag.fit(frame, list(FINAL_FEATURES) + extra, target)
            bag.save(out / name)
            how = "fitted"
        else:
            shutil.copytree(parent_models / name, out / name)
            how = f"copied from {PARENT}"
        pieces[name] = {"target": target, "features": len(FINAL_FEATURES) + len(extra), "extra_inputs": extra, "how": how, "fingerprint": h.fingerprint(out / name)}
        print(f"[{VERSION}] {name}: {how}, fingerprint {pieces[name]['fingerprint'][:16]}", flush=True)
    return {"version": VERSION, "parent": PARENT, "training_rows": len(train), "last_training_day": str(train[DATE_COL].max().date()), "training_features_sha256": sha, "pieces": pieces}


MODELS_DIR = v1.MODELS_DIR.parent / VERSION


def fit_v1_11(log_repo: Path, training_features: Path = v1.TRAINING_FEATURES) -> dict:
    """The real fit: fit the two Saturday pieces into models/v1.11, copy v1's other pieces beside them, and register the version in the log repository's `models/versions.json` (append-only).
    Refuses if v1.11 is already registered or its folder is not empty, if v1 is not registered, if v1.11's correction history is missing or differs from v1's (same cut-off, so the same file),
    or if the checkout has modified tracked files (a version is fitted from a clean, recorded commit)."""
    versions_path = log_repo / "models" / "versions.json"
    registry = json.loads(versions_path.read_text(encoding="utf-8"))
    if any(v["version"] == VERSION for v in registry["versions"]):
        raise RuntimeError(f"version {VERSION} is already registered in {versions_path}; a refit must be a new version")
    parent = next((v for v in registry["versions"] if v["version"] == PARENT), None)
    if parent is None:
        raise RuntimeError(f"the parent version {PARENT} is not registered in {versions_path}")
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        raise RuntimeError(f"{MODELS_DIR} is not empty; models are fitted once and never overwritten")
    history = v1.correction_history_path(VERSION)
    if not history.exists():
        raise RuntimeError(f"{history} is missing: v1.11 has v1's cut-off, so it uses v1's correction history; copy data/training/{PARENT}_correction_history.csv to that name")
    if h.file_sha256(history) != parent["correction_history_sha256"]:
        raise RuntimeError(f"{history} is not the correction history v1 recorded")
    commit = v1.code_commit()
    summary = fit_pieces(training_features, MODELS_DIR)
    for name, spec in summary["pieces"].items():
        if spec["how"].startswith("copied") and spec["fingerprint"] != parent["pieces"][name]["fingerprint"]:
            raise RuntimeError(f"the copied piece {name} does not match the fingerprint v1 registered; {MODELS_DIR} must be removed before another attempt")
    pieces = {name: {"target": spec["target"], "features": spec["features"], "extra_inputs": spec["extra_inputs"], "how": spec["how"],
                     "seeds": len(h.Bag.load(MODELS_DIR / name).models), "fingerprint": spec["fingerprint"]} for name, spec in summary["pieces"].items()}
    entry = {
        "version": VERSION, "parent": PARENT, "fitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "first_excluded_day": parent["first_excluded_day"], "last_training_day": summary["last_training_day"], "training_rows": summary["training_rows"],
        "training_features_sha256": summary["training_features_sha256"], "correction_history_sha256": h.file_sha256(history), "code_commit": commit,
        "weekend_options": parent["weekend_options"], "correction": parent["correction"], "arms": [ARM],
        "change_from_parent": f"{INTENSITY} (src/features/payday_intensity.py) added, last, to the Saturday pieces {', '.join(SATURDAY_PIECES)}; the other pieces are {PARENT}'s own files",
        "pieces": pieces,
        "note": "CatBoost writes a random id into every saved model, so these fingerprints identify THESE files; a refit is a new version.",
    }
    registry["versions"].append(entry)
    versions_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"[{VERSION}] registered in {versions_path}", flush=True)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("fit-pieces")
    dry.add_argument("--training-features", type=Path, required=True)
    dry.add_argument("--out", type=Path, required=True)
    real = sub.add_parser("fit", help="the real fit and registration (after the pre-registration amendment)")
    real.add_argument("--log-repo", type=Path, required=True)
    real.add_argument("--training-features", type=Path, default=v1.TRAINING_FEATURES)
    args = parser.parse_args()
    if args.command == "fit":
        fit_v1_11(args.log_repo, args.training_features)
        return
    summary = fit_pieces(args.training_features, args.out)
    print(f"[{VERSION}] dry run only: {len(summary['pieces'])} pieces in {args.out}; training rows {summary['training_rows']} to {summary['last_training_day']}; NOT registered anywhere")


if __name__ == "__main__":
    main()
