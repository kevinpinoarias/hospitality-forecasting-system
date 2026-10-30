"""
Version v1 of the models behind the forecast log (the private repository `forecasting-log`; rules in its PREREGISTRATION.md).

The log has three arms: the portfolio model (the frozen deployed model), the post model (the weekend package) and the hydro model (the post model plus the Thursday-Hydro flag).
Version v1 is fitted ONCE on the rows dated before 7 September 2026 (the holdout test's cut-off), from the frozen engineered-features file, then saved and fingerprinted. It is never refitted:
a refit is a new version. The pieces are:

  log             the log model: Monday to Thursday of the post model, and the Wednesday-to-Thursday base of both package arms
  log_hydro_thu   the log model with the Thursday-Hydro flag as an extra input: Monday to Thursday of the hydro model
  raw             the plain raw-pound model: the Saturday fallback when the two Saturday lags are not known
  fri_A_all       Friday: log target, inputs A_all (sales on the same weekday 1 to 4 weeks ago)
  sat_A_lags      Saturday: raw-pound target, inputs A_lags (sales 2 and 3 weeks ago)
  sun_B           Sunday: log target, inputs B (weekend interactions)

The portfolio model is not refitted: its files (models/final_model) are verified against the fingerprint named in the holdout pre-registration.

Run (from the runner folder):
  python -m src.forecast_log.v1 export-history --out data/training/v1_correction_history.csv      (in the development folder, where the backtest results exist)
  python -m src.forecast_log.v1 fit --log-repo D:\\Projectos\\DRG\\forecasting-log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import CONFIG, PROJECT_ROOT
from src.experiments import run_series35_holdout as h
from src.features.events import concert_calendar
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

VERSION = "v1"
FIRST_EXCLUDED_DAY = pd.Timestamp("2026-09-07")  # the first day NOT in the training data
TRAINING_FEATURES = PROJECT_ROOT / "data" / "training" / "v1_engineered_features.csv"
CORRECTION_HISTORY = PROJECT_ROOT / "data" / "training" / "v1_correction_history.csv"
MODELS_DIR = PROJECT_ROOT / "models" / VERSION
THURSDAY = 3
WEEKEND_PIECES = {"fri_A_all": "Friday", "sat_A_lags": "Saturday", "sun_B": "Sunday"}
# name -> (extra inputs on top of the 38 final-model features, target)
PIECES: dict[str, tuple[list[str], str]] = {
    "log": ([], "log1p"),
    "log_hydro_thu": (["hydro_thu"], "log1p"),
    "raw": ([], "raw"),
    "fri_A_all": (h.EXTRA["Friday"], "log1p"),
    "sat_A_lags": (h.EXTRA["Saturday"], "raw"),
    "sun_B": (h.EXTRA["Sunday"], "log1p"),
}


def add_hydro_thu(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `hydro_thu` (a Hydro show on a Thursday) from the curated concert list. The list must cover the dates asked for."""
    cal = concert_calendar(str(frame[DATE_COL].min().date()), str(frame[DATE_COL].max().date()))
    flag = (cal["hydro_show"] * (cal[DATE_COL].dt.dayofweek == THURSDAY)).rename("hydro_thu")
    out = frame.drop(columns=["hydro_thu"], errors="ignore").merge(pd.concat([cal[DATE_COL], flag], axis=1), on=DATE_COL, how="left")
    out["hydro_thu"] = out["hydro_thu"].fillna(0).astype(int)
    return out


def training_frames() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The v1 training rows (plain, and with the weekend inputs added) and the hash of the features file they come from."""
    sha = h.file_sha256(TRAINING_FEATURES)
    if sha != h.FROZEN_FEATURES_SHA:
        raise RuntimeError(f"{TRAINING_FEATURES} is not the frozen features file (sha256 {sha[:16]}..., expected {h.FROZEN_FEATURES_SHA[:16]}...)")
    full = add_hydro_thu(h.load_engineered(TRAINING_FEATURES))
    train = h.complete_rows(full)
    train = train[train[DATE_COL] < FIRST_EXCLUDED_DAY].reset_index(drop=True)
    train_w = h.with_weekend(train, full.set_index(DATE_COL)[TARGET_COL])
    return train, train_w, sha


def code_commit() -> str:
    """The commit of the code doing the fit; refuses a checkout with modified tracked files (untracked data files are fine)."""
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=PROJECT_ROOT).returncode != 0:
        raise RuntimeError("tracked files in this checkout are modified: a version is fitted from a clean, recorded commit")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def verify_portfolio_model() -> str:
    got = h.fingerprint(CONFIG.path("final_model_dir"))
    if got != h.FROZEN_PORTFOLIO_SHA:
        raise RuntimeError(f"models/final_model is not the frozen portfolio model (sha256 {got[:16]}..., expected {h.FROZEN_PORTFOLIO_SHA[:16]}...)")
    return got


def fit_v1(log_repo: Path) -> dict:
    versions_path = log_repo / "models" / "versions.json"
    registry = json.loads(versions_path.read_text(encoding="utf-8"))
    if any(v["version"] == VERSION for v in registry["versions"]):
        raise RuntimeError(f"version {VERSION} is already registered in {versions_path}; a refit must be a new version")
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        raise RuntimeError(f"{MODELS_DIR} is not empty; models are fitted once and never overwritten")
    commit = code_commit()
    portfolio = verify_portfolio_model()
    if not CORRECTION_HISTORY.exists():
        raise RuntimeError(f"{CORRECTION_HISTORY} is missing (export it in the development folder with `export-history`)")
    train, train_w, features_sha = training_frames()
    print(f"[v1] training rows {len(train)} ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}); Hydro Thursdays in training: {int(train['hydro_thu'].sum())}", flush=True)
    pieces = {}
    for name, (extra, target) in PIECES.items():
        frame = train_w if name in WEEKEND_PIECES else train
        bag = h.Bag.fit(frame, list(FINAL_FEATURES) + extra, target)
        bag.save(MODELS_DIR / name)
        pieces[name] = {"target": target, "features": len(FINAL_FEATURES) + len(extra), "extra_inputs": extra, "seeds": len(bag.models),
                        "fingerprint": h.fingerprint(MODELS_DIR / name)}
        print(f"[v1] {name}: {len(bag.models)} seeds, fingerprint {pieces[name]['fingerprint'][:16]}", flush=True)
    entry = {
        "version": VERSION, "fitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "first_excluded_day": str(FIRST_EXCLUDED_DAY.date()), "last_training_day": str(train[DATE_COL].max().date()), "training_rows": len(train),
        "training_features_sha256": features_sha, "correction_history_sha256": h.file_sha256(CORRECTION_HISTORY), "code_commit": commit,
        "weekend_options": h.CHOSEN, "correction": {k: list(v) if isinstance(v, tuple) else v for k, v in h.CORRECTION.items()},
        "portfolio_model": {"fingerprint": portfolio, "source": "models/final_model (frozen deployed model)"},
        "pieces": pieces,
        "note": "CatBoost writes a random id into every saved model, so these fingerprints identify THESE files; a refit is a new version.",
    }
    registry["versions"].append(entry)
    registry["note"] = "append-only: one entry per fitted version; an entry is never edited"
    versions_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"[v1] registered in {versions_path}", flush=True)
    return entry


def export_correction_history(out: Path) -> None:
    """The backtest days and the post model's own out-of-sample weekend forecasts for them, which the Saturday and Sunday correction learns from
    (frozen at the cut-off). Needs the stored backtest results, so it is run in the development folder and the file is copied to the runner."""
    days, weekend = h.package_history(FIRST_EXCLUDED_DAY)
    frame = days[["date", "fold_id", "actual", "weekday"]].copy()
    frame["package_weekend_forecast"] = weekend
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[v1] correction history: {len(frame)} days ({frame['date'].min()} to {frame['date'].max()}) -> {out}; sha256 {h.file_sha256(out)[:16]}")


def load_correction_history(path: Path = CORRECTION_HISTORY) -> tuple[pd.DataFrame, "pd.Series"]:
    frame = pd.read_csv(path)
    return frame[["date", "fold_id", "actual", "weekday"]], frame["package_weekend_forecast"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--log-repo", type=Path, required=True)
    export = sub.add_parser("export-history")
    export.add_argument("--out", type=Path, default=CORRECTION_HISTORY)
    args = parser.parse_args()
    if args.command == "fit":
        fit_v1(args.log_repo)
    else:
        export_correction_history(args.out)


if __name__ == "__main__":
    main()
