"""
Version v1.13 of the models behind the forecast log: v1.12 with the year-end specialist models.

What it is (agreed 9 November 2026, after Series 79 to 92 in docs/EXPERIMENT_LOG.md and the plan in docs/JANUARY_DECEMBER_CANDIDATES_2026_11.md):
  * the hydro arm of v1.12, with its five pieces (`log_hydro_thu`, `fri_A_all`, `sat_A_lags`, `raw`, `sun_B`) copied UNCHANGED as v1.12's own files, plus two further sets of the same five pieces used on two stretches only:
      - the COMPANION on 26 to 31 December (`comp_<piece>`): each piece refitted WITHOUT the ten recent-sales inputs (they carry Christmas into the days after it: the venue is shut or slow, and "last week" is not a guide);
      - the MASKED model on 1 to 8 January (`mask_<piece>`): each piece refitted with the ten inputs set to missing on every training row dated 1 to 8 January, and set to missing on the forecast days of that stretch.
    Every other day of the year is v1.12's own forecast, so v1.13 differs from v1.12 on 14 days a year (26 December to 8 January) and Q7 answers one question.
  * the ten inputs: lag_7_sales, rolling_7_sales, rolling_14_sales, lag_1_fe, lag_7_fe, rolling_7_fe (in the 38 final-model features) and lag_14_sales, lag_21_sales, same_wd_mean4, same_wd_max4 (the Friday and
    Saturday pieces' extras); a piece only has the ones it uses. This is the definition of Series 82 (j1_8) and Series 85 (the companion), the series the backtest result comes from.
  * the same training data and cut-off as v1 to v1.12 (the rows before 2 November 2026 of the frozen features file), the same correction history, the same weekend options.
  * the specialist pieces are saved beside the main ones in models/v1.13 as `comp_<piece>` and `mask_<piece>`; the routing is in src/forecast_log/predict.py (`with_year_end_specialists`).
  The rows of the hydro arm in forecasts.csv have model `hydro_v1.13` and model version `v1.13`.

Dry run (writes only to the folder given):
  python -m src.forecast_log.v1_13 fit-pieces --training-features <frozen v1_engineered_features.csv> --out <folder>
The real fit and registration (clean checkout; the frozen features file is in the runner folder, not the development folder):
  python -m src.forecast_log.v1_13 fit --log-repo <path-to>\\forecasting-log --training-features <frozen v1_engineered_features.csv>
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import run_series35_holdout as h
from src.forecast_core.routing import COMPANION_PREFIX, COMPANION_WINDOW, INPUTS, MASK_PREFIX, MASK_WINDOW, in_window  # noqa: F401  (moved there unchanged)
from src.forecast_log import v1, v1_12
from src.models.final_model import DATE_COL, FINAL_FEATURES

VERSION = "v1.13"
PARENT = v1_12.VERSION
ARM = "hydro_v1.13"
MAIN_PIECES = v1_12.HYDRO_ARM_PIECES
MODELS_DIR = v1.MODELS_DIR.parent / VERSION


def base_features(name: str) -> list[str]:
    """The inputs of a main piece (v1.12's own): the 38 final-model features plus the piece's extras."""
    return list(FINAL_FEATURES) + v1_12.PIECES[name][0]


def piece_features(kind: str, name: str) -> list[str]:
    """The inputs of a piece of one kind: `main`, `companion` (without the ten recent-sales inputs) or `masked` (all of them, missing inside the window)."""
    base = base_features(name)
    return [f for f in base if f not in INPUTS] if kind == "companion" else base


def specialist_names() -> dict[str, tuple[str, str]]:
    """Every fitted specialist piece: file name -> (kind, main piece it is the specialist of)."""
    return {**{COMPANION_PREFIX + n: ("companion", n) for n in MAIN_PIECES}, **{MASK_PREFIX + n: ("masked", n) for n in MAIN_PIECES}}


def masked_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of `frame` with the ten inputs set to missing on the rows dated inside the mask window (only the ones the frame has)."""
    out = frame.copy()
    window = in_window(out[DATE_COL], *MASK_WINDOW)
    for c in INPUTS:
        if c in out.columns:
            col = out[c].astype(float).to_numpy().copy()
            col[window] = np.nan
            out[c] = col
    return out


def fit_pieces(training_features: Path, out: Path, parent_models: Path | None = None) -> dict:
    """Dry run: copy v1.12's five pieces unchanged into `out` and fit the ten specialist pieces on the v1 training rows (the companion without the ten inputs, the masked model with them missing
    on the rows of 1 to 8 January). Nothing is registered."""
    parent_models = parent_models or v1_12.MODELS_DIR
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty; models are fitted once and never overwritten")
    train, train_w, sha = v1_12.training_frames(training_features)
    masked, masked_w = masked_copy(train), masked_copy(train_w)
    pieces = {}
    for name in MAIN_PIECES:
        shutil.copytree(parent_models / name, out / name)
        pieces[name] = {"kind": "main", "target": v1_12.PIECES[name][1], "features": len(base_features(name)), "extra_inputs": v1_12.PIECES[name][0], "how": f"copied from {PARENT}", "fingerprint": h.fingerprint(out / name)}
        print(f"[{VERSION}] {name}: copied from {PARENT}, fingerprint {pieces[name]['fingerprint'][:16]}", flush=True)
    for file, (kind, name) in specialist_names().items():
        weekend = name in v1.WEEKEND_PIECES
        frame = (masked_w if weekend else masked) if kind == "masked" else (train_w if weekend else train)
        features = piece_features(kind, name)
        bag = h.Bag.fit(frame, features, v1_12.PIECES[name][1])
        bag.save(out / file)
        pieces[file] = {"kind": kind, "target": v1_12.PIECES[name][1], "features": len(features), "extra_inputs": [f for f in features if f not in FINAL_FEATURES], "how": "fitted", "fingerprint": h.fingerprint(out / file)}
        print(f"[{VERSION}] {file}: fitted ({kind}, {len(features)} inputs), fingerprint {pieces[file]['fingerprint'][:16]}", flush=True)
    return {"version": VERSION, "parent": PARENT, "training_rows": len(train), "last_training_day": str(train[DATE_COL].max().date()), "training_features_sha256": sha, "pieces": pieces}


def fit_v1_13(log_repo: Path, training_features: Path = v1.TRAINING_FEATURES) -> dict:
    """The real fit: copy v1.12's five pieces into models/v1.13, fit the ten specialist pieces beside them, and register the version in the log repository's `models/versions.json` (append-only).
    Refuses if v1.13 is already registered or its folder is not empty, if v1.12 is not registered, if v1.13's correction history is missing or differs from the parent's (same cut-off, so the same file),
    if a copied piece does not match the fingerprint its parent registered, or if the checkout has modified tracked files (a version is fitted from a clean, recorded commit)."""
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
        raise RuntimeError(f"{history} is missing: {VERSION} has {PARENT}'s cut-off, so it uses {PARENT}'s correction history; copy data/training/{PARENT}_correction_history.csv to that name")
    if h.file_sha256(history) != parent["correction_history_sha256"]:
        raise RuntimeError(f"{history} is not the correction history {PARENT} recorded")
    commit = v1.code_commit()
    summary = fit_pieces(training_features, MODELS_DIR)
    for name, spec in summary["pieces"].items():
        if spec["how"].startswith("copied") and spec["fingerprint"] != parent["pieces"][name]["fingerprint"]:
            raise RuntimeError(f"the copied piece {name} does not match the fingerprint {PARENT} registered; {MODELS_DIR} must be removed before another attempt")
    pieces = {name: {"kind": spec["kind"], "target": spec["target"], "features": spec["features"], "extra_inputs": spec["extra_inputs"], "how": spec["how"],
                     "seeds": len(h.Bag.load(MODELS_DIR / name).models), "fingerprint": spec["fingerprint"]} for name, spec in summary["pieces"].items()}
    entry = {
        "version": VERSION, "parent": PARENT, "fitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "first_excluded_day": parent["first_excluded_day"], "last_training_day": summary["last_training_day"], "training_rows": summary["training_rows"],
        "training_features_sha256": summary["training_features_sha256"], "correction_history_sha256": h.file_sha256(history), "code_commit": commit,
        "weekend_options": parent["weekend_options"], "correction": parent["correction"], "arms": [ARM],
        "change_from_parent": f"two sets of the five hydro-arm pieces used on two stretches only (src/forecast_log/predict.py, with_year_end_specialists): the companion pieces ({COMPANION_PREFIX}<piece>), fitted without the ten "
                              f"recent-sales inputs, on 26 to 31 December; the masked pieces ({MASK_PREFIX}<piece>), fitted with the ten inputs missing on the training rows of 1 to 8 January, on 1 to 8 January with the ten inputs "
                              f"set to missing; the five main pieces are {PARENT}'s own files, so {VERSION} differs from {PARENT} on 26 December to 8 January only",
        "specialist_windows": {"companion": "26 December to 31 December", "masked": "1 January to 8 January"},
        "recent_sales_inputs": INPUTS,
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
    dry.add_argument("--parent-models", type=Path, default=None, help="the parent version's model folder (default: models/<parent>); a dry run of the parent can stand in for it")
    real = sub.add_parser("fit", help="the real fit and registration (after the pre-registration amendment)")
    real.add_argument("--log-repo", type=Path, required=True)
    real.add_argument("--training-features", type=Path, default=v1.TRAINING_FEATURES)
    args = parser.parse_args()
    if args.command == "fit":
        fit_v1_13(args.log_repo, args.training_features)
        return
    summary = fit_pieces(args.training_features, args.out, args.parent_models)
    print(f"[{VERSION}] dry run only: {len(summary['pieces'])} pieces in {args.out}; training rows {summary['training_rows']} to {summary['last_training_day']}; NOT registered anywhere")


if __name__ == "__main__":
    main()
