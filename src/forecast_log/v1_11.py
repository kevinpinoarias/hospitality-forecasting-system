"""
Version v1.11 of the models behind the forecast log: v1 with the triangle payday intensity added to the two Saturday pieces.

STATUS (7 November 2026): DEFINED BUT NOT FITTED, NOT REGISTERED AND NOT RUN IN THE LOG. Kevin decided the version's content on 7 November and that it is not to be included in the forecasts for now. Nothing here writes to the
log repository, `models/versions.json`, the runner's configuration or `config/forecast_log.yaml`. `fit_pieces` fits into a folder given on the command line (a dry run). Registering the version needs, first, an amendment
to the log's PREREGISTRATION.md saying how v1.11 is scored (rule 1: how a later version is scored is written before its retrain), and Kevin's go-ahead.

What it is (decided 7 November 2026, after Series 70-72 in docs/EXPERIMENT_LOG.md):
  * the hydro model plus `intensity_sym` (src/features/payday_intensity.py), in the Saturday pieces ONLY: `sat_A_lags` (Saturday with its two lags) and `raw` (the Saturday fallback). The Monday-to-Thursday,
    Friday and Sunday pieces get nothing new: the intensity's gain was only consistent on Saturday (Saturday -20.8 pounds a day, interval -35.9 to -6.2; every other weekday's interval includes zero and its sign flips between 2025 and 2026).
  * the same training data as v1: the rows before 2 November 2026 of the frozen features file, so the only difference from v1 is the new column.
  * the pieces that did not change are v1's own files, copied unchanged (same fingerprints), so v1 and v1.11 forecast identically on every day except Saturdays. The hydro arm needs five pieces:
    `log_hydro_thu`, `fri_A_all`, `sat_A_lags`, `raw` and `sun_B`. (The `log` piece belongs to the post arm only and is not part of v1.11.) The portfolio model is the frozen deployed model and is not part of it.
  * the new column is appended LAST to the two Saturday pieces' inputs (column order is part of a fit's definition).

Dry run (writes only to the folder given):
  python -m src.forecast_log.v1_11 fit-pieces --training-features <frozen v1_engineered_features.csv> --out <folder>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.experiments import run_series35_holdout as h
from src.features.payday_intensity import COLUMN as INTENSITY
from src.features.payday_intensity import add_payday_intensity
from src.forecast_log import v1
from src.models.final_model import DATE_COL, FINAL_FEATURES

VERSION = "v1.11"
PARENT = v1.VERSION
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-pieces")
    fit.add_argument("--training-features", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = fit_pieces(args.training_features, args.out)
    print(f"[{VERSION}] dry run only: {len(summary['pieces'])} pieces in {args.out}; training rows {summary['training_rows']} to {summary['last_training_day']}; NOT registered anywhere")


if __name__ == "__main__":
    main()
