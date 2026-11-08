"""
Version v1.12 of the models behind the forecast log: v1.11 with the closed-day fix in the intensity, plus the holiday rule.

What it is (decided 8 November 2026, after Series 76-78 in docs/EXPERIMENT_LOG.md):
  * the hydro arm of v1.11, with the two Saturday pieces (`sat_A_lags` and `raw`) refitted on `intensity_fixed` (src/features/payday_intensity.py) instead of `intensity_sym`: the same triangle payday
    intensity, but 0 on 25 December and 1 January (the venue is shut: zero sales on those days in the data) and -0.5 on 26 December (Boxing Day: open but slow). December's pay dates are unchanged: Series 76-78
    tried an early pay date (the 20th, the Friday of its week, the 27th, the 15th dropped, a larger penalty) and none of it beat the fix alone.
  * the same training data and cut-off as v1 and v1.11 (the rows before 2 November 2026 of the frozen features file), so the only difference from v1.11 is the Saturday pieces' intensity column.
  * the three other hydro-arm pieces (`log_hydro_thu`, `fri_A_all`, `sun_B`) are v1.11's own files (which are v1's own), copied unchanged, so v1.12 and v1.11 forecast identically on every day except Saturdays.
  * the new column is appended LAST to the two Saturday pieces' inputs (column order is part of a fit's definition), in the place of `intensity_sym`.
  * the holiday rule (src/forecast_log/holiday_rule.py, version r1) belongs to v1.12 and is issued with it: 0 on 25 December and 1 January, the earlier Boxing Days' mean on 26 December, as rows of the
    model `holiday_rule` beside the models' own rows. It is not a fitted model and has nothing to register; the `holiday_rule` setting switches it on.
  The rows of the hydro arm in forecasts.csv have model `hydro_v1.12` and model version `v1.12` (the log's de-duplication key has no version column, so each version's arm needs a name of its own).

Dry run (writes only to the folder given):
  python -m src.forecast_log.v1_12 fit-pieces --training-features <frozen v1_engineered_features.csv> --out <folder>
The real fit and registration (clean checkout; the frozen features file is in the runner folder, not the development folder):
  python -m src.forecast_log.v1_12 fit --log-repo D:\\Projectos\\DRG\\forecasting-log --training-features <frozen v1_engineered_features.csv>
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiments import run_series35_holdout as h
from src.features.payday_intensity import COLUMN_FIXED as INTENSITY
from src.features.payday_intensity import add_payday_intensity_fixed
from src.forecast_log import v1, v1_11
from src.models.final_model import DATE_COL, FINAL_FEATURES

VERSION = "v1.12"
PARENT = v1_11.VERSION
ARM = "hydro_v1.12"
SATURDAY_PIECES = v1_11.SATURDAY_PIECES
HYDRO_ARM_PIECES = v1_11.HYDRO_ARM_PIECES
# name -> (extra inputs on top of the 38 final-model features, target); only the Saturday pieces differ from v1.11 (and v1), and there the closed-day column takes the place of the intensity
PIECES: dict[str, tuple[list[str], str]] = {
    name: ((v1.PIECES[name][0] + [INTENSITY]) if name in SATURDAY_PIECES else v1.PIECES[name][0], v1.PIECES[name][1]) for name in HYDRO_ARM_PIECES
}
MODELS_DIR = v1.MODELS_DIR.parent / VERSION


def training_frames(training_features: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The v1 training rows (plain, and with the weekend inputs added), each with the closed-day intensity column, from the FROZEN features file (its hash is checked as v1's is)."""
    sha = h.file_sha256(training_features)
    if sha != v1.TRAINING_FEATURES_SHA:
        raise RuntimeError(f"{training_features} is not the frozen features file (sha256 {sha[:16]}..., expected {v1.TRAINING_FEATURES_SHA[:16]}...)")
    full = v1.add_hydro_thu(h.load_engineered(training_features))
    train = h.complete_rows(full)
    train = train[train[DATE_COL] < v1.FIRST_EXCLUDED_DAY].reset_index(drop=True)
    train_w = h.with_weekend(train, full.set_index(DATE_COL)[v1.TARGET_COL])
    return add_payday_intensity_fixed(train, DATE_COL), add_payday_intensity_fixed(train_w, DATE_COL), sha


def fit_pieces(training_features: Path, out: Path, parent_models: Path | None = None) -> dict:
    """Dry run: fit the two Saturday pieces on the v1 training rows with the closed-day intensity, and copy v1.11's other three hydro-arm pieces unchanged, into `out`. Nothing is registered."""
    parent_models = parent_models or v1_11.MODELS_DIR
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


def fit_v1_12(log_repo: Path, training_features: Path = v1.TRAINING_FEATURES) -> dict:
    """The real fit: fit the two Saturday pieces into models/v1.12, copy v1.11's other pieces beside them, and register the version in the log repository's `models/versions.json` (append-only).
    Refuses if v1.12 is already registered or its folder is not empty, if v1.11 is not registered, if v1.12's correction history is missing or differs from the parent's (same cut-off, so the same file),
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
    pieces = {name: {"target": spec["target"], "features": spec["features"], "extra_inputs": spec["extra_inputs"], "how": spec["how"],
                     "seeds": len(h.Bag.load(MODELS_DIR / name).models), "fingerprint": spec["fingerprint"]} for name, spec in summary["pieces"].items()}
    entry = {
        "version": VERSION, "parent": PARENT, "fitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "first_excluded_day": parent["first_excluded_day"], "last_training_day": summary["last_training_day"], "training_rows": summary["training_rows"],
        "training_features_sha256": summary["training_features_sha256"], "correction_history_sha256": h.file_sha256(history), "code_commit": commit,
        "weekend_options": parent["weekend_options"], "correction": parent["correction"], "arms": [ARM],
        "change_from_parent": f"{INTENSITY} (src/features/payday_intensity.py: intensity_sym with 25 December and 1 January at 0 and 26 December at -0.5) takes the place of intensity_sym, last, in the "
                              f"Saturday pieces {', '.join(SATURDAY_PIECES)}; the other pieces are {PARENT}'s own files; the holiday rule (src/forecast_log/holiday_rule.py, r1) is issued with the version and is not fitted",
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
        fit_v1_12(args.log_repo, args.training_features)
        return
    summary = fit_pieces(args.training_features, args.out)
    print(f"[{VERSION}] dry run only: {len(summary['pieces'])} pieces in {args.out}; training rows {summary['training_rows']} to {summary['last_training_day']}; NOT registered anywhere")


if __name__ == "__main__":
    main()
