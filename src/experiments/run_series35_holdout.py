"""
Series 35 - the holdout test on the new venue days (7 September to 15 October 2026).

The rules are in docs/HOLDOUT_2026_10_PREREGISTRATION.md, written and committed before any new figure was opened. This script is
the machinery. Three stages, run in order, each as its own command:

  prepare   Anonymise the new exports and rebuild the features in an ISOLATED copy of the pipeline (data/holdout/), so nothing
            in data/raw, data/features, models or reports changes.
  fit       Fit every model on rows dated BEFORE the first holdout day only, and fingerprint the model files. Committed before
            scoring so the models cannot have been adjusted after seeing the new days.
  score     Forecast the holdout days in the three modes (A operational, B blind from the cut-off, C Monday rota run for the
            following week), with and without the manager's forecast as an input, for every arm, and write the tables.

Arms: M the manager's forecast, S a seasonal baseline, P the portfolio (deployed) model on the original school calendar,
D the product final model on the verified calendar, K the Track A package.

Rehearsal: `--cutoff 2026-06-25` runs the same code on OLD days (the last backtest block onward) so the machinery can be tested
without touching the new data. In a rehearsal the portfolio arm is refitted (no frozen model exists for that cut-off) and the
package's correction history is cut at the cut-off.

Run:  python -m src.experiments.run_series35_holdout prepare
      python -m src.experiments.run_series35_holdout fit   [--cutoff 2026-09-07]
      python -m src.experiments.run_series35_holdout score [--cutoff 2026-09-07]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import CONFIG, PROJECT_ROOT
from src.experiments import run_package_validation as pv
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series26_weekend_features as s26
from src.experiments import run_series29_school_calendar as s29
from src.experiments.series19_21_common import load_protocol, test_days
from src.features.build_features import LONG_BREAK_TYPES, SCHOOL_BREAKS_V1
from src.features.serving_features import (
    WEATHER_COLS, build_feature_frame, extend_daily_frame, fill_future_sales_seasonally, seasonal_medians, with_history_features,
)
from src.models.final_model import FINAL_FEATURES, FINAL_PARAMS, BaggedCatBoost

DATE_COL, TARGET_COL = "date", "total_sales"
REAL_CUTOFF = pd.Timestamp("2026-09-07")  # the first holdout day; everything before it is the frozen past
SEEDS = list(range(25))
HOLDOUT_DIR = PROJECT_ROOT / "data" / "holdout"
HOLDOUT_CONFIG = PROJECT_ROOT / "config" / "holdout.yaml"
LOG_DIR = PROJECT_ROOT / "docs" / "holdout_log"
FROZEN_PORTFOLIO_SHA = "b7d347460573956e55fb7f655d5e4d60a17cda5a327ba2c4ec8ec7b4bbbb924c"  # models/final_model, see the pre-registration
FROZEN_FEATURES_SHA = "ad16ffd7aad53f09fb66a22ec62fbc0d3562d1127e9ac0ab69476daf4d9c64b0"
CORRECTION = pv.CORRECTION
CHOSEN = {"Friday": "A_all", "Saturday": "A_lags", "Sunday": "B"}  # Series 27 / 31 on the verified calendar, fixed in the pre-registration
EXTRA = {"Friday": s26.VARIANTS["A_all"]["extra"], "Saturday": s26.VARIANTS["A_lags"]["extra"], "Sunday": s26.VARIANTS["B"]["extra"]}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VARIANTS = ("with_manager", "no_manager")
ARM_LABELS = {"M": "Manager's forecast", "S": "Seasonal baseline", "P": "Portfolio model (deployed)",
              "D": "Product model (verified calendar)", "K": "Track A package"}


# --------------------------------------------------------------------------- helpers: bags, files


class Bag:
    """A CatBoost bag over seeds on one target (log1p or raw pounds); the final model's own recipe."""

    def __init__(self, models: list, features: list[str], target: str) -> None:
        self.models, self.features, self.target = models, list(features), target

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list[str], target: str, seeds: list[int] = SEEDS, thread_count: int = 4) -> "Bag":
        from catboost import CatBoostRegressor

        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        models = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=thread_count, **FINAL_PARAMS)
            model.fit(train[list(features)], y)
            models.append(model)
        return cls(models, features, target)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(X[self.features]) for m in self.models]
        if self.target == "log1p":
            preds = [np.expm1(p) for p in preds]
        return np.mean(preds, axis=0)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("seed_*.cbm"):
            old.unlink()
        for i, m in enumerate(self.models):
            m.save_model(str(directory / f"seed_{i:02d}.cbm"))
        (directory / "meta.json").write_text(json.dumps({"features": self.features, "target": self.target, "seeds": len(self.models)}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Bag":
        from catboost import CatBoostRegressor

        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        models = []
        for i in range(meta["seeds"]):
            m = CatBoostRegressor()
            m.load_model(str(directory / f"seed_{i:02d}.cbm"))
            models.append(m)
        return cls(models, meta["features"], meta["target"])


def fingerprint(directory: Path) -> str:
    """sha256 of every file in a folder, sorted by name and concatenated (the pre-registration's convention)."""
    h = hashlib.sha256()
    for f in sorted(p for p in directory.iterdir() if p.is_file()):
        h.update(f.read_bytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_engineered(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)


def complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with every final-model feature and the target present (the protocol's model frame)."""
    required = sorted(set(FINAL_FEATURES) | {DATE_COL, TARGET_COL})
    return df.dropna(subset=required).reset_index(drop=True)


def with_weekend(rows: pd.DataFrame, sales_by_date: pd.Series) -> pd.DataFrame:
    return s26.add_weekend_features(rows, sales_by_date)


def v1_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with its four school columns recomputed from the ORIGINAL school table (what the deployed model was trained on)."""
    return s29.with_calendar(frame, SCHOOL_BREAKS_V1, LONG_BREAK_TYPES)


def weekday_names(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(list(dates))).dt.day_name().to_numpy()


def stand_in_manager(history: pd.DataFrame, dates) -> np.ndarray:
    """What the live API uses when no manager forecast is given: the weekday-and-season median of the manager's past forecasts."""
    return seasonal_medians(history, list(dates), "forecast_sales")


def monday_runs(first_day: pd.Timestamp, last_day: pd.Timestamp) -> list[pd.Timestamp]:
    """Issue Mondays whose following-week TARGET week (Monday+7 to Monday+13) starts on or after the holdout's first day and has at least
    one day with an actual. The first holdout week has no run (its issue day would fall in the frozen past)."""
    out, m = [], first_day + pd.Timedelta(days=(0 - first_day.weekday()) % 7)
    while m + pd.Timedelta(days=7) <= last_day:
        if m >= first_day:
            out.append(m)
        m += pd.Timedelta(days=7)
    return out


# --------------------------------------------------------------------------- stage 1: prepare


def write_holdout_config() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    h = "data/holdout"
    cfg["paths"].update({
        "raw_data_dir": f"{h}/raw", "interim_dir": f"{h}/interim", "interim_sales_dir": f"{h}/interim/sales",
        "interim_labour_dir": f"{h}/interim/labour", "processed_sales_dir": f"{h}/processed/sales",
        "processed_labour_dir": f"{h}/processed/labour",
        "daily_sales_master": f"{h}/processed/sales/daily_sales_totals_master.csv",
        "daily_labour_master": f"{h}/processed/labour/daily_labour_totals_master.csv",
        "features_dir": f"{h}/features", "engineered_features": f"{h}/features/engineered_features.csv",
        "cache_dir": f"{h}/data_cache/open_meteo", "results_dir": f"{h}/reports/results", "figures_dir": f"{h}/reports/figures",
        "experiments_dir": f"{h}/reports/experiments", "models_dir": f"{h}/models", "final_model_dir": f"{h}/models/final_model",
    })
    header = ("# The isolated copy of the pipeline's paths for the holdout test (Series 35): same settings as config.yaml, every\n"
              "# generated path under data/holdout/ (gitignored). raw_private_dir is the real landing folder for the new exports.\n")
    HOLDOUT_CONFIG.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def prepare() -> None:
    write_holdout_config()
    raw = HOLDOUT_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not any(raw.glob("*.csv")):
        for f in sorted((PROJECT_ROOT / "data" / "raw").glob("*.csv")):
            shutil.copy2(f, raw / f.name)
    env = {**__import__("os").environ, "FORECAST_CONFIG": str(HOLDOUT_CONFIG), "PYTHONPATH": str(PROJECT_ROOT), "WANDB_MODE": "disabled"}
    steps = [
        "from src.ingestion.anonymize_raw import anonymize_raw; anonymize_raw()",
        "from src.preprocessing.split_sales_labour import split_sales_and_labour; split_sales_and_labour()",
        "from src.preprocessing.build_daily_sales import build_daily_sales; build_daily_sales()",
        "from src.preprocessing.build_daily_labour import build_daily_labour; build_daily_labour()",
        "from src.features.build_features import build_feature_dataset; build_feature_dataset()",
    ]
    for code in steps:
        print(f"[prepare] {code.split(';')[0].split(' import ')[1] if ' import ' in code else code}", flush=True)
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"prepare step failed:\n{code}\n{done.stdout[-1500:]}\n{done.stderr[-1500:]}")
    new = load_engineered(HOLDOUT_DIR / "features" / "engineered_features.csv")
    old = load_engineered(CONFIG.path("engineered_features"))
    print(f"[prepare] holdout features: {len(new)} days, {new[DATE_COL].min().date()} to {new[DATE_COL].max().date()} "
          f"(frozen file: {len(old)} days to {old[DATE_COL].max().date()})")
    shared = new[new[DATE_COL] <= old[DATE_COL].max()].set_index(DATE_COL)
    frozen = old.set_index(DATE_COL)
    cols = [c for c in FINAL_FEATURES if c in shared.columns]
    diff = (shared[cols] - frozen.loc[shared.index, cols]).abs().max()
    print("[prepare] largest difference from the frozen features on the shared days, per feature (weather revisions are expected):")
    print(diff[diff > 1e-9].round(4).to_string() if (diff > 1e-9).any() else "  none: identical")


# --------------------------------------------------------------------------- stage 2: fit


def frozen_features_path() -> Path:
    return CONFIG.path("engineered_features")


def fit(cutoff: pd.Timestamp, models_dir: Path) -> dict:
    real = cutoff == REAL_CUTOFF
    path = frozen_features_path()
    if real and file_sha256(path) != FROZEN_FEATURES_SHA:
        raise RuntimeError("the frozen engineered features have changed since the pre-registration; refusing to fit")
    full = load_engineered(path)
    rows = complete_rows(full)
    train = rows[rows[DATE_COL] < cutoff]
    train_w = with_weekend(train, full.set_index(DATE_COL)[TARGET_COL])
    print(f"[fit] cut-off {cutoff.date()}: {len(train)} training rows ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()})", flush=True)
    specs = {
        "D": (train, FINAL_FEATURES, "log1p"),
        "raw": (train, FINAL_FEATURES, "raw"),
        "Fri_A_all": (train_w, FINAL_FEATURES + EXTRA["Friday"], "log1p"),
        "Sat_A_lags": (train_w, FINAL_FEATURES + EXTRA["Saturday"], "raw"),
        "Sun_B": (train_w, FINAL_FEATURES + EXTRA["Sunday"], "log1p"),
        "P_fresh": (v1_calendar(train), FINAL_FEATURES, "log1p"),
    }
    prints = {}
    for name, (frame, features, target) in specs.items():
        bag = Bag.fit(frame, features, target)
        bag.save(models_dir / name)
        prints[name] = fingerprint(models_dir / name)
        print(f"[fit] {name}: {len(bag.models)} seeds, sha256 {prints[name][:16]}", flush=True)
    if real:
        got = fingerprint(CONFIG.path("final_model_dir"))
        if got != FROZEN_PORTFOLIO_SHA:
            raise RuntimeError("models/final_model is not the frozen portfolio model named in the pre-registration")
        prints["P_frozen (models/final_model)"] = got
    manifest = {"cutoff": str(cutoff.date()), "rows": len(train), "last_training_day": str(train[DATE_COL].max().date()),
                "features_sha256": file_sha256(path), "fingerprints": prints}
    (models_dir / "fingerprints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if real:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "fit_fingerprints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[fit] fingerprints written to {LOG_DIR / 'fit_fingerprints.json'} - commit them BEFORE scoring")
    return manifest


def load_arms(cutoff: pd.Timestamp, models_dir: Path) -> dict[str, Bag]:
    arms = {n: Bag.load(models_dir / n) for n in ("D", "raw", "Fri_A_all", "Sat_A_lags", "Sun_B", "P_fresh")}
    if cutoff == REAL_CUTOFF:  # the portfolio arm is the frozen deployed model, not the refit
        b, _ = BaggedCatBoost.load(CONFIG.path("final_model_dir"))
        arms["P"] = Bag(b.models, b.features, "log1p")
    else:
        arms["P"] = arms["P_fresh"]
    return arms


# --------------------------------------------------------------------------- stage 3: score - the package


def package_history(cutoff: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    """The 585-day backtest days (cut at `cutoff` in a rehearsal) with the validated package's own out-of-sample forecasts for them
    (verified-calendar results, seeds 0-24): the error history the Saturday and Sunday correction learns from."""
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    data = pv.load_set("_cal2")
    chosen = pv.choose_on_first_blocks(days, data["log_raw_0_24"], data["features_0_24"])
    if chosen != CHOSEN:
        raise RuntimeError(f"the weekend options re-derived from the results ({chosen}) differ from the pre-registered ones ({CHOSEN})")
    weekend = pv.package_steps(days, data["log_raw_0_24"], data["features_0_24"], chosen)[pv.STEPS[2]]
    keep = pd.to_datetime(days["date"]).to_numpy() < np.datetime64(cutoff)
    return days[keep].reset_index(drop=True), weekend[keep]


def correct_new_block(hist_days: pd.DataFrame, hist_pred: np.ndarray, new_dates, new_pred: np.ndarray) -> np.ndarray:
    """The Series 24 correction for a new test block: shift each Saturday and Sunday forecast by -strength x the recency-weighted bias of the
    package's errors in the history, exactly as the protocol does for a block after the history."""
    new_days = pd.DataFrame({"date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in new_dates],
                             "fold_id": int(hist_days["fold_id"].max()) + 1, "actual": np.nan, "weekday": weekday_names(new_dates)})
    days = pd.concat([hist_days[["date", "fold_id", "actual", "weekday"]], new_days], ignore_index=True)
    pred = np.concatenate([hist_pred, new_pred])
    return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])[len(hist_days):]


def package_mode_a(arms: dict[str, Bag], rows: pd.DataFrame, sales: pd.Series) -> np.ndarray:
    """The package as validated: every weekend day with its chosen option, Monday to Thursday the log model."""
    w = with_weekend(rows, sales)
    wd = weekday_names(rows[DATE_COL])
    out = arms["D"].predict(rows)
    out = np.where(wd == "Friday", arms["Fri_A_all"].predict(w), out)
    out = np.where(wd == "Saturday", arms["Sat_A_lags"].predict(w), out)
    return np.where(wd == "Sunday", arms["Sun_B"].predict(w), out)


def package_served(arms: dict[str, Bag], rows: pd.DataFrame, sales_known: pd.Series) -> np.ndarray:
    """The package's routing at distance (pre-registration section 5): Monday to Friday the plain log model, Saturday the raw model with
    A_lags where both lags are known (plain raw otherwise), Sunday the log model with option B."""
    w = with_weekend(rows, sales_known)
    wd = weekday_names(rows[DATE_COL])
    out = arms["D"].predict(rows)
    lags_known = w[EXTRA["Saturday"]].notna().all(axis=1).to_numpy()
    saturday = np.where(lags_known, arms["Sat_A_lags"].predict(w), arms["raw"].predict(rows))
    out = np.where(wd == "Saturday", saturday, out)
    return np.where(wd == "Sunday", arms["Sun_B"].predict(w), out)


# --------------------------------------------------------------------------- stage 3: score - the three modes


def build_frame(daily: pd.DataFrame, calendar: str) -> pd.DataFrame:
    frame = build_feature_frame(daily)
    return v1_calendar(frame) if calendar == "v1" else frame


def served_rows(full: pd.DataFrame, last_known_day: pd.Timestamp, target_dates, calendar: str = "v2",
                manager: np.ndarray | None = None, observed_weather: bool = False) -> pd.DataFrame:
    """Feature rows for `target_dates`, built the way the live API builds a date beyond its data: real history up to `last_known_day`, the
    days after it filled with weekday/season medians (sales, manager forecast, weather), the model's own pipeline functions for every
    calendar and weather feature. Nothing dated after `last_known_day` is read except, if asked, the observed weather."""
    history = full[full[DATE_COL] <= last_known_day]
    end = max(pd.Timestamp(d) for d in target_dates)
    daily = extend_daily_frame(history, end.date())
    if observed_weather:
        real = full.set_index(DATE_COL)
        later = daily[DATE_COL] > last_known_day
        have = daily.loc[later, DATE_COL].isin(real.index)
        idx = daily.index[later][have.to_numpy()]
        for col in WEATHER_COLS:
            daily.loc[idx, col] = real.loc[daily.loc[idx, DATE_COL], col].to_numpy()
    frame = build_frame(daily, calendar)
    first_future = len(history)
    position = {d: i for i, d in enumerate(frame[DATE_COL])}
    idx = [position[pd.Timestamp(d)] for d in target_dates]
    sales = fill_future_sales_seasonally(frame, first_future, history)
    rows = with_history_features(frame, sales, idx)
    if manager is not None:
        rows["forecast_sales"] = manager
    return rows


def sales_known_series(full: pd.DataFrame, last_known_day: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = full[full[DATE_COL] <= last_known_day].set_index(DATE_COL)[TARGET_COL]
    return s.reindex(pd.date_range(s.index.min(), end, freq="D"))


def predictions_mode_a(arms: dict[str, Bag], full: pd.DataFrame, cutoff: pd.Timestamp, hist_days, hist_pred) -> pd.DataFrame:
    hold = full[(full[DATE_COL] >= cutoff)].dropna(subset=FINAL_FEATURES).reset_index(drop=True)
    sales = full.set_index(DATE_COL)[TARGET_COL]
    out = []
    for variant in VARIANTS:
        rows = hold.copy()
        if variant == "no_manager":
            rows["forecast_sales"] = [stand_in_manager(full[full[DATE_COL] < d], [d])[0] for d in rows[DATE_COL]]
        pk = package_mode_a(arms, rows, sales)
        pk = correct_new_block(hist_days, hist_pred, rows[DATE_COL], pk)
        for arm, pred in (("P", arms["P"].predict(v1_calendar(rows))), ("D", arms["D"].predict(rows)), ("K", pk)):
            out.append(pd.DataFrame({"mode": "A", "issue": rows[DATE_COL] - pd.Timedelta(days=1), "last_known": rows[DATE_COL] - pd.Timedelta(days=1),
                                     "date": rows[DATE_COL], "arm": arm, "variant": variant, "forecast": pred}))
    return pd.concat(out, ignore_index=True)


def served_predictions(arms, full, last_known_day, targets, mode, issue, hist_days, hist_pred, observed_weather=False) -> pd.DataFrame:
    targets = list(targets)
    real_manager = full.set_index(DATE_COL)["forecast_sales"].reindex(targets).to_numpy()
    sales_k = sales_known_series(full, last_known_day, max(targets))
    out = []
    for variant in VARIANTS:
        manager = real_manager if variant == "with_manager" else None
        rows_v2 = served_rows(full, last_known_day, targets, "v2", manager, observed_weather)
        rows_v1 = served_rows(full, last_known_day, targets, "v1", manager, observed_weather)
        pk = correct_new_block(hist_days, hist_pred, targets, package_served(arms, rows_v2, sales_k))
        for arm, pred in (("P", arms["P"].predict(rows_v1)), ("D", arms["D"].predict(rows_v2)), ("K", pk)):
            out.append(pd.DataFrame({"mode": mode, "issue": issue, "last_known": last_known_day, "date": targets, "arm": arm,
                                     "variant": variant, "forecast": pred}))
    return pd.concat(out, ignore_index=True)


def baselines(full: pd.DataFrame, pred_frame: pd.DataFrame) -> pd.DataFrame:
    """Arms M and S for every (mode, issue, date) in `pred_frame`."""
    keys = pred_frame[["mode", "issue", "last_known", "date"]].drop_duplicates()
    manager = full.set_index(DATE_COL)["forecast_sales"]
    rows = []
    for _, k in keys.iterrows():
        s = seasonal_medians(full[full[DATE_COL] <= k["last_known"]], [k["date"]], TARGET_COL)[0]  # only days known when the forecast is made
        for variant in VARIANTS:
            base = {"mode": k["mode"], "issue": k["issue"], "last_known": k["last_known"], "date": k["date"], "variant": variant}
            rows.append({**base, "arm": "M", "forecast": manager.get(k["date"], np.nan)})
            rows.append({**base, "arm": "S", "forecast": s})
    return pd.DataFrame(rows)


def score(cutoff: pd.Timestamp, models_dir: Path, out_dir: Path, features_path: Path) -> pd.DataFrame:
    arms = load_arms(cutoff, models_dir)
    full = load_engineered(features_path)
    last_actual = full.dropna(subset=[TARGET_COL])[DATE_COL].max()
    hist_days, hist_pred = package_history(cutoff)
    holdout_days = list(full[(full[DATE_COL] >= cutoff) & (full[DATE_COL] <= last_actual)][DATE_COL])
    print(f"[score] holdout {holdout_days[0].date()} to {holdout_days[-1].date()} ({len(holdout_days)} days)", flush=True)
    parts = [predictions_mode_a(arms, full, cutoff, hist_days, hist_pred)]
    parts.append(served_predictions(arms, full, cutoff - pd.Timedelta(days=1), holdout_days, "B", cutoff - pd.Timedelta(days=1), hist_days, hist_pred))
    for m in monday_runs(cutoff, last_actual):
        targets = [m + pd.Timedelta(days=7 + i) for i in range(7)]
        parts.append(served_predictions(arms, full, m - pd.Timedelta(days=1), targets, "C", m, hist_days, hist_pred))
        parts.append(served_predictions(arms, full, m - pd.Timedelta(days=1), targets, "C_observed_weather", m, hist_days, hist_pred, observed_weather=True))
        print(f"[score] Monday run {m.date()} done", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    pred = pd.concat([pred, baselines(full, pred)], ignore_index=True)
    actual = full.set_index(DATE_COL)[TARGET_COL]
    pred["actual"] = pred["date"].map(actual)
    pred["manager"] = pred["date"].map(full.set_index(DATE_COL)["forecast_sales"])
    pred = pred[pred["actual"].notna()].reset_index(drop=True)
    pred["weekday"] = pred["date"].dt.day_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "holdout_predictions.csv", index=False)
    print(f"[score] {len(pred)} scored forecasts written to {out_dir / 'holdout_predictions.csv'}", flush=True)
    return pred


# --------------------------------------------------------------------------- stage 3: score - the tables


def block_gain(a: np.ndarray, b: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """Mean of |a error| - |b error| (positive = b was closer) with a 95% moving-block bootstrap interval, 7-day blocks."""
    diff = np.abs(a - actual) - np.abs(b - actual)
    return s23.masked_block_bootstrap(diff, np.ones(len(diff)), block=min(7, len(diff)))


def cell(pred: pd.DataFrame, mode: str, arm: str, variant: str) -> pd.DataFrame:
    sub = pred[(pred["mode"] == mode) & (pred["arm"] == arm) & (pred["variant"] == variant)]
    return sub.sort_values("date").reset_index(drop=True)


def metrics(sub: pd.DataFrame) -> dict:
    err = sub["forecast"] - sub["actual"]
    fri_sun = sub["weekday"].isin(["Friday", "Saturday", "Sunday"])
    return {"days": len(sub), "MAE": float(err.abs().mean()), "bias": float(err.mean()),
            "FriSun_MAE": float(err[fri_sun].abs().mean()) if fri_sun.any() else np.nan,
            "MonThu_MAE": float(err[~fri_sun].abs().mean()) if (~fri_sun).any() else np.nan}


def summary_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, arm, variant), sub in pred.groupby(["mode", "arm", "variant"]):
        rows.append({"mode": mode, "arm": arm, "variant": variant, **metrics(sub)})
    return pd.DataFrame(rows)


def q1_label(gain: tuple[float, float, float], model_mae: float) -> str:
    mean, lo, hi = gain
    if model_mae > 1333:
        return "Drift alarm (model MAE above the historical 95th percentile of 39-day windows, GBP 1,333)"
    if lo > 0:
        return "Confirmed (the 95% interval is entirely above zero)"
    if mean > 0:
        return "Consistent (better than the manager on average; the interval includes zero, as expected with this little data)"
    return "Warning (no better than the manager on average; about 1 historical window in 12 does this)"


def package_label(change_all: float, change_fri_sun: float) -> str:
    if change_all > 30:
        return "contradicted (worse than the product final model by more than GBP 30 a day)"
    if change_all < 0 and change_fri_sun < 0:
        return "directionally consistent (better on all days and on Friday to Sunday; not promoted on this window)"
    return "no signal (mixed)"


# --------------------------------------------------------------------------- reports


def weekly_mae(pred: pd.DataFrame, mode: str, variant: str) -> pd.DataFrame:
    sub = pred[(pred["mode"] == mode) & (pred["variant"] == variant) & pred["arm"].isin(["M", "P", "D", "K"])].copy()
    sub["week_starting"] = sub["date"] - pd.to_timedelta(sub["date"].dt.dayofweek, unit="D")
    sub["abs_err"] = (sub["forecast"] - sub["actual"]).abs()
    table = sub.pivot_table(index="week_starting", columns="arm", values="abs_err", aggfunc="mean")
    return table[[a for a in ("M", "P", "D", "K") if a in table.columns]]


def simple_report(pred: pd.DataFrame, out_dir: Path, rehearsal: bool) -> str:
    def mae(mode, arm, variant):
        c = cell(pred, mode, arm, variant)
        return metrics(c) if len(c) else None

    a_m, a_p = cell(pred, "A", "M", "with_manager"), cell(pred, "A", "P", "with_manager")
    gain = block_gain(a_m["forecast"].to_numpy(), a_p["forecast"].to_numpy(), a_m["actual"].to_numpy())
    p_mae = metrics(a_p)["MAE"]
    d_all = cell(pred, "A", "D", "with_manager")
    k_all = cell(pred, "A", "K", "with_manager")
    k_chg = (np.abs(k_all["forecast"] - k_all["actual"]).mean() - np.abs(d_all["forecast"] - d_all["actual"]).mean())
    fs = k_all["weekday"].isin(["Friday", "Saturday", "Sunday"]).to_numpy()
    k_fs = (np.abs(k_all["forecast"] - k_all["actual"])[fs].mean() - np.abs(d_all["forecast"] - d_all["actual"])[fs].mean())

    lines = []
    tag = "REHEARSAL on old days (not the real holdout)" if rehearsal else "Holdout: new venue days"
    lines.append(f"# {tag}\n")
    lines.append(f"Days scored in the daily (operational) test: {len(a_p)} ({a_p['date'].min().date()} to {a_p['date'].max().date()}).\n")
    lines.append("## The main question: is the deployed model still better than the manager?\n")
    lines.append(f"- Manager's average miss: **GBP {metrics(a_m)['MAE']:,.0f}** a day")
    lines.append(f"- Deployed model's average miss: **GBP {p_mae:,.0f}** a day")
    lines.append(f"- Gap in the model's favour: **GBP {gain[0]:+,.0f}** a day (95% interval {gain[1]:+,.0f} to {gain[2]:+,.0f})")
    lines.append(f"- Verdict (the pre-registered rule): **{q1_label(gain, p_mae)}**\n")

    lines.append("## Average miss in pounds a day (lower is better)\n")
    lines.append("| | Daily, with manager | Daily, without | Monday run, with manager | Monday run, without |")
    lines.append("|---|---|---|---|---|")
    for arm in ("M", "P", "D", "K"):
        cells = []
        for mode, variant in (("A", "with_manager"), ("A", "no_manager"), ("C", "with_manager"), ("C", "no_manager")):
            m = mae(mode, arm, variant)
            cells.append("n/a" if m is None else f"{m['MAE']:,.0f} (bias {m['bias']:+,.0f})")
        lines.append(f"| {ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
    lines.append("\nBias = forecast minus actual, so a minus sign means the forecast ran low. The manager's own forecast does not change between the "
                 "'with' and 'without' columns; only what the models are given does.\n")
    lines.append(f"The package against the product final model (daily): **{k_chg:+,.0f}** a day on all days, **{k_fs:+,.0f}** on Friday to Sunday "
                 f"({package_label(k_chg, k_fs)}).\n")

    wk = weekly_mae(pred, "A", "with_manager").round(0)
    lines.append("## Week by week (daily test, with manager): average miss in pounds\n")
    lines.append("| Week starting | " + " | ".join(ARM_LABELS[c] for c in wk.columns) + " |")
    lines.append("|---|" + "---|" * len(wk.columns))
    for ts, row in wk.iterrows():
        lines.append(f"| {ts.date()} | " + " | ".join(f"{v:,.0f}" for v in row) + " |")
    return "\n".join(lines)


def full_report(pred: pd.DataFrame) -> str:
    s = summary_table(pred).round(1)
    out = ["# Full results: every mode, arm and variant\n", s.to_markdown(index=False), ""]
    for mode in ("A", "B", "C"):
        for variant in VARIANTS:
            m_ = cell(pred, mode, "M", variant)
            if not len(m_):
                continue
            out.append(f"\n## Paired gain over the manager, mode {mode}, {variant} (positive = the arm was closer; 95% interval)\n")
            out.append("| Arm | Days | Gain | Low | High |\n|---|---|---|---|---|")
            for arm in ("S", "P", "D", "K"):
                c = cell(pred, mode, arm, variant)
                if len(c) != len(m_):
                    continue
                g = block_gain(m_["forecast"].to_numpy(), c["forecast"].to_numpy(), m_["actual"].to_numpy())
                out.append(f"| {ARM_LABELS[arm]} | {len(c)} | {g[0]:+,.0f} | {g[1]:+,.0f} | {g[2]:+,.0f} |")
    for mode in ("A", "B", "C"):
        w = weekly_mae(pred, mode, "with_manager").round(0)
        out.append(f"\n## Week by week, mode {mode}, with manager\n\n" + w.to_markdown())
    out.append("\n## Context: mode C with observed weather in place of the fallback\n")
    s2 = summary_table(pred[pred["mode"] == "C_observed_weather"]).round(1)
    out.append(s2.to_markdown(index=False) if len(s2) else "(none)")
    return "\n".join(out)


def draw_chart(pred: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wk = weekly_mae(pred, "A", "with_manager")
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = {"M": "#777777", "P": "#1a5fb4", "D": "#2f8f4e", "K": "#c64600"}
    for arm in wk.columns:
        ax.plot(wk.index, wk[arm], marker="o", label=ARM_LABELS[arm], color=colours[arm], lw=2 if arm in ("M", "P") else 1.3)
    ax.set_ylabel("average miss, GBP a day")
    ax.set_title("Weekly average miss, daily test (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "fit", "score"])
    parser.add_argument("--cutoff", default=str(REAL_CUTOFF.date()))
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    rehearsal = cutoff != REAL_CUTOFF
    models_dir = HOLDOUT_DIR / ("models" if not rehearsal else f"models_rehearsal_{cutoff.date()}")
    out_dir = HOLDOUT_DIR / "reports" / ("results" if not rehearsal else f"rehearsal_{cutoff.date()}")
    if args.stage == "prepare":
        prepare()
    elif args.stage == "fit":
        fit(cutoff, models_dir)
    else:
        features = frozen_features_path() if rehearsal else HOLDOUT_DIR / "features" / "engineered_features.csv"
        pred = score(cutoff, models_dir, out_dir, features)
        (out_dir / "results_simple.md").write_text(simple_report(pred, out_dir, rehearsal), encoding="utf-8")
        (out_dir / "results_full.md").write_text(full_report(pred), encoding="utf-8")
        draw_chart(pred, out_dir / "weekly_miss.png")
        print(f"[score] reports written to {out_dir}")


if __name__ == "__main__":
    main()
