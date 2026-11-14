"""
Version v1.14 of the models behind the forecast log: v1.13 with the year-ago sales input in three pieces.

What it is (agreed 9 November 2026, after Series 90 to 94 in docs/EXPERIMENT_LOG.md and the plan in docs/JANUARY_DECEMBER_CANDIDATES_2026_11.md):
  * the hydro arm of v1.13 with one new input, `sales_364` (src/features/year_ago.py: the sales on the same weekday 364 days earlier), appended LAST to the inputs of three pieces: the Monday-to-Thursday piece with
    the hydro flag (`log_hydro_thu`, which also serves Friday when the Friday inputs are not known), the Friday piece (`fri_A_all`) and the Saturday piece (`sat_A_lags`). These are the three pieces of the
    "Monday to Saturday" form of Series 93 (-8.03 and -9.73 pounds a day against v1.13 on 100 and fresh seeds); Series 94 found no better form of the input.
  * the Sunday piece (`sun_B`) is NOT given the input (Series 92 and 93: Sundays get worse, +11 to +14 a day; leaving it out is the default, not a proven necessity), and neither is the raw Saturday
    fallback (`raw`, used only when the Saturday lags are not known; it was not part of the tested form). Both are v1.13's own files, copied unchanged.
  * the ten specialist pieces of v1.13 (the companion on 26 to 31 December, the masked model on 1 to 8 January) are v1.13's own files, copied unchanged, without the input: the test composed the package that way
    (Series 93), and the input does not move late December. On those 14 days v1.14 therefore forecasts what v1.13 forecasts, apart from the weekend correction.
  * the same training data and cut-off as v1 to v1.13 (the rows before 2 November 2026 of the frozen features file); 655 of the 1,006 training rows have a year-ago value (none before 2025; a missing value is
    read by the trees as their own value); the same correction history and weekend options.
  The rows of the hydro arm in forecasts.csv have model `hydro_v1.14` and model version `v1.14`.

Dry run (writes only to the folder given):
  python -m src.forecast_log.v1_14 fit-pieces --training-features <frozen v1_engineered_features.csv> --out <folder>
The real fit and registration (clean checkout; the frozen features file is in the runner folder, not the development folder):
  python -m src.forecast_log.v1_14 fit --log-repo D:\\Projectos\\DRG\\forecasting-log --training-features <frozen v1_engineered_features.csv>
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.experiments import run_series35_holdout as h
from src.features.year_ago import COLUMN as YEAR_AGO
from src.features.year_ago import add_year_ago_sales
from src.forecast_log import v1, v1_12, v1_13
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

VERSION = "v1.14"
PARENT = v1_13.VERSION
ARM = "hydro_v1.14"
YEAR_AGO_PIECES = ("log_hydro_thu", "fri_A_all", "sat_A_lags")  # Monday to Thursday (hydro), Friday, Saturday
MODELS_DIR = v1.MODELS_DIR.parent / VERSION


def piece_inputs(name: str) -> list[str]:
    """The inputs of a main piece: the 38 final-model features plus the piece's extras, and `sales_364` last in the three pieces that take it."""
    return v1_13.base_features(name) + ([YEAR_AGO] if name in YEAR_AGO_PIECES else [])


def training_frames(training_features: Path):
    """The v1 training rows (plain, and with the weekend inputs added), each with the closed-day intensity and `sales_364`, from the FROZEN features file (its hash is checked as v1's is)."""
    train, train_w, sha = v1_12.training_frames(training_features)
    sales = h.load_engineered(training_features).set_index(DATE_COL)[TARGET_COL]
    return add_year_ago_sales(train, sales, DATE_COL), add_year_ago_sales(train_w, sales, DATE_COL), sha


def fit_pieces(training_features: Path, out: Path, parent_models: Path | None = None) -> dict:
    """Dry run: fit the three pieces on the v1 training rows with `sales_364` and copy v1.13's other twelve unchanged into `out`. Nothing is registered."""
    parent_models = parent_models or v1_13.MODELS_DIR
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty; models are fitted once and never overwritten")
    train, train_w, sha = training_frames(training_features)
    pieces = {}
    for name in [*v1_13.MAIN_PIECES, *v1_13.specialist_names()]:
        kind = "main" if name in v1_13.MAIN_PIECES else v1_13.specialist_names()[name][0]
        target = v1_12.PIECES[name][1] if name in v1_13.MAIN_PIECES else v1_12.PIECES[v1_13.specialist_names()[name][1]][1]
        if name in YEAR_AGO_PIECES:
            features = piece_inputs(name)
            bag = h.Bag.fit(train_w if name in v1.WEEKEND_PIECES else train, features, target)
            bag.save(out / name)
            how = "fitted"
        else:
            shutil.copytree(parent_models / name, out / name)
            features = h.Bag.load(out / name).features
            how = f"copied from {PARENT}"
        pieces[name] = {"kind": kind, "target": target, "features": len(features), "extra_inputs": [f for f in features if f not in FINAL_FEATURES], "how": how, "fingerprint": h.fingerprint(out / name)}
        print(f"[{VERSION}] {name}: {how}, {len(features)} inputs, fingerprint {pieces[name]['fingerprint'][:16]}", flush=True)
    return {"version": VERSION, "parent": PARENT, "training_rows": len(train), "last_training_day": str(train[DATE_COL].max().date()), "training_features_sha256": sha,
            "rows_with_year_ago": int(train[YEAR_AGO].notna().sum()), "pieces": pieces}


def fit_v1_14(log_repo: Path, training_features: Path = v1.TRAINING_FEATURES) -> dict:
    """The real fit: fit the three pieces into models/v1.14, copy v1.13's other twelve beside them, and register the version in the log repository's `models/versions.json` (append-only).
    Refuses if v1.14 is already registered or its folder is not empty, if v1.13 is not registered, if v1.14's correction history is missing or differs from the parent's (same cut-off, so the same file),
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
        "change_from_parent": f"{YEAR_AGO} (src/features/year_ago.py: the sales on the same weekday 364 days earlier) is appended last to the inputs of {', '.join(YEAR_AGO_PIECES)}; the other twelve pieces "
                              f"(the Sunday piece, the raw Saturday fallback and the ten year-end specialist pieces) are {PARENT}'s own files",
        "specialist_windows": parent["specialist_windows"],
        "recent_sales_inputs": parent["recent_sales_inputs"],
        "training_rows_with_year_ago": summary["rows_with_year_ago"],
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
        fit_v1_14(args.log_repo, args.training_features)
        return
    summary = fit_pieces(args.training_features, args.out, args.parent_models)
    print(f"[{VERSION}] dry run only: {len(summary['pieces'])} pieces in {args.out}; training rows {summary['training_rows']} to {summary['last_training_day']} ({summary['rows_with_year_ago']} with a year-ago value); NOT registered anywhere")


if __name__ == "__main__":
    main()
