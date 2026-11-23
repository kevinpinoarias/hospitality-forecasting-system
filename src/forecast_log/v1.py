"""
Version v1 of the models behind the forecast log (the private repository `forecasting-log`; rules in its PREREGISTRATION.md).

The log has three arms: the portfolio model (the frozen deployed model), the post model (the weekend package) and the hydro model (the post model plus the Thursday-Hydro flag).
Version v1 is fitted ONCE on the rows dated before 2 November 2026 (the last day of the export available on 1 November), from a frozen copy of the engineered features, then saved
and fingerprinted. It is never refitted: a refit is a new version. (An earlier fit to 6 September, made on 30 October, was withdrawn before any forecast was logged: the log's
PREREGISTRATION.md, amendment of 1 November 2026.) The pieces are:

  log             the log model: Monday to Thursday of the post model, and the Wednesday-to-Thursday base of both package arms
  log_hydro_thu   the log model with the Thursday-Hydro flag as an extra input: Monday to Thursday of the hydro model
  raw             the plain raw-pound model: the Saturday fallback when the two Saturday lags are not known
  fri_A_all       Friday: log target, inputs A_all (sales on the same weekday 1 to 4 weeks ago)
  sat_A_lags      Saturday: raw-pound target, inputs A_lags (sales 2 and 3 weeks ago)
  sun_B           Sunday: log target, inputs B (weekend interactions)

The portfolio model is not refitted: its files (models/final_model) are verified against the fingerprint named in the holdout pre-registration.

Run on Product-branch (the development folder), then copy models/v1 and data/training/v1_correction_history.csv to the runner:
  python -m src.forecast_log.v1 export-history --withdrawn-models <folder of the withdrawn fit's six pieces>
  python -m src.forecast_log.v1 fit --log-repo <path-to>\\forecasting-log
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import CONFIG, PROJECT_ROOT
from src.forecast_core.correction import correction_history_path  # noqa: F401  (moved there unchanged)
from src.forecast_core.rows import THURSDAY, add_hydro_thu  # noqa: F401
from src.experiments import run_series35_holdout as h
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

VERSION = "v1"
FIRST_EXCLUDED_DAY = pd.Timestamp("2026-11-02")  # the first day NOT in the training data
BACKTEST_END = pd.Timestamp("2026-09-07")  # the backtest's days end before this; the withdrawn fit forecast the days from here to the cut-off
TRAINING_FEATURES = PROJECT_ROOT / "data" / "training" / "v1_engineered_features.csv"
TRAINING_FEATURES_SHA = "c5355ae36638b5180a232f04992276821bc26bb797c6c7833be8c00eb6b5733a"  # data/features/engineered_features.csv as rebuilt on 2 November 2026


CORRECTION_HISTORY = correction_history_path(VERSION)
MODELS_DIR = PROJECT_ROOT / "models" / VERSION
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


def training_frames() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The v1 training rows (plain, and with the weekend inputs added) and the hash of the features file they come from."""
    sha = h.file_sha256(TRAINING_FEATURES)
    if sha != TRAINING_FEATURES_SHA:
        raise RuntimeError(f"{TRAINING_FEATURES} is not the frozen features file (sha256 {sha[:16]}..., expected {TRAINING_FEATURES_SHA[:16]}...)")
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


def withdrawn_fit_days(withdrawn_models: Path, next_fold: int) -> pd.DataFrame:
    """The days from the end of the backtest to the cut-off, with the withdrawn fit's forecasts for them (trained before 7 September, so out of
    sample): the package as validated, each weekend day with its chosen option and the real inputs, as the backtest's own forecasts were made."""
    arms = {"D": h.Bag.load(withdrawn_models / "log"), "Fri_A_all": h.Bag.load(withdrawn_models / "fri_A_all"),
            "Sat_A_lags": h.Bag.load(withdrawn_models / "sat_A_lags"), "Sun_B": h.Bag.load(withdrawn_models / "sun_B")}
    full = h.load_engineered(TRAINING_FEATURES)
    rows = h.complete_rows(full)
    rows = rows[(rows[DATE_COL] >= BACKTEST_END) & (rows[DATE_COL] < FIRST_EXCLUDED_DAY)].reset_index(drop=True)
    expected = pd.date_range(BACKTEST_END, FIRST_EXCLUDED_DAY - pd.Timedelta(days=1), freq="D")
    if not rows[DATE_COL].reset_index(drop=True).equals(pd.Series(expected, name=DATE_COL)):
        raise RuntimeError(f"the training features do not hold every day from {BACKTEST_END.date()} to {(FIRST_EXCLUDED_DAY - pd.Timedelta(days=1)).date()}")
    frame = pd.DataFrame({"date": rows[DATE_COL].dt.strftime("%Y-%m-%d"), "fold_id": next_fold, "actual": rows[TARGET_COL].to_numpy(),
                          "weekday": h.weekday_names(rows[DATE_COL])})
    frame["package_weekend_forecast"] = h.package_mode_a(arms, rows, full.set_index(DATE_COL)[TARGET_COL])
    return frame


def export_correction_history(out: Path, withdrawn_models: Path) -> None:
    """The days and the post model's own out-of-sample forecasts for them, which the Saturday and Sunday correction learns from (frozen at the
    cut-off): the backtest's days (to 6 September), then the withdrawn fit's forecasts to the cut-off as one more block. Needs the stored backtest
    results, so it is run in the development folder and the file is copied to the runner."""
    days, weekend = h.package_history(BACKTEST_END)
    frame = days[["date", "fold_id", "actual", "weekday"]].copy()
    frame["package_weekend_forecast"] = weekend
    frame = pd.concat([frame, withdrawn_fit_days(withdrawn_models, int(frame["fold_id"].max()) + 1)], ignore_index=True)
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
    export.add_argument("--withdrawn-models", type=Path, required=True, help="the folder holding the withdrawn fit's six pieces (log, fri_A_all, sat_A_lags, sun_B used)")
    args = parser.parse_args()
    if args.command == "fit":
        fit_v1(args.log_repo)
    else:
        export_correction_history(args.out, args.withdrawn_models)


if __name__ == "__main__":
    main()
