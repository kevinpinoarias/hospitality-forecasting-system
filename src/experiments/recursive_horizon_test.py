"""
Recursive multi-day-ahead horizon test (Series 3).

Series 2 (src/experiments/run_feature_sweep.py) found that short-lag
sales/forecast-error history features are the strongest lever found in
this project so far. Those features are not target leakage - every one
of them only ever references strictly historical, already-observed days
(see docs/EXPERIMENT_LOG.md's Series 2 write-up for the verification). But
Series 2's rolling-origin evaluation always had access to the TRUE
actuals for the days immediately preceding whatever date it was scoring
- an assumption that only holds for a forecast generated shortly before
its target date, with fresh data. It does NOT hold for the kind of
longer-horizon forecast this project's own API explicitly supports
(`days_beyond_training_data` exists precisely to flag exactly this kind
of trust gap) - for a date three weeks out, the "sales 7 days before it"
haven't happened yet at the time you'd generate that forecast.

This experiment tests the honest version of that scenario: fit the model
ONCE per fold (as normal), then predict forward day by day through the
test window, feeding each lag/rolling feature from the model's OWN prior
predictions wherever the referenced date falls inside the batch being
forecast, rather than from real history. Reports how much the lag
features' advantage decays at increasing forecast horizons (day 1, day 7,
day 30 of a single forward run), compared against the baseline 15-feature
set (which has no lag/rolling dependency at all, and is therefore
horizon-invariant by construction - included as the reference line).

Outputs
-------
- reports/experiments/recursive_horizon_results.csv (day-by-day predictions)
- reports/experiments/recursive_horizon_summary.csv (MAE at day 1/7/30)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.adapters import DATE_COL, FEATURES as BASELINE_FEATURES, TARGET_COL
from src.experiments.feature_sets import ALL_CANDIDATES
from src.experiments.run_feature_sweep import ENGINEERED_FEATURES_PATH, MODEL_PARAMS, build_folds, load_data
from src.experiments.splits import Fold, slice_train_window

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments"


def load_full_history() -> pd.DataFrame:
    """
    Load date/total_sales/forecast_sales from the FULL engineered feature
    dataset (979 rows, no calendar-row gaps), not the feature-selection
    dataset used elsewhere in this sweep (950 rows, strict-dropna'd across
    every candidate column - which drops 15 mid-series dates entirely
    because some UNRELATED feature, e.g. weather, was occasionally
    missing there). The recursion buffer only needs real sales/forecast
    history to be complete; it has no reason to inherit gaps caused by a
    completely different column's occasional missing value.

    One single real, pre-existing data gap remains even here: total_sales
    is genuinely missing for 2025-01-24 (the same date model_features.csv
    silently drops via its own dropna elsewhere in this project - a real
    reporting gap in the underlying export, not a code bug). Left as NaN,
    this one missing day would propagate into every lag/rolling feature
    that references it for up to 14 days afterwards, breaking recursion
    buffer continuity for reasons that have nothing to do with what this
    experiment is testing. It is linearly interpolated here - explicitly
    logged, not silently patched - purely so the recursion has a
    continuous buffer to read from; it does not affect which dates are
    actually scored.
    """
    df = pd.read_csv(ENGINEERED_FEATURES_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df[[DATE_COL, TARGET_COL, "forecast_sales"]].sort_values(DATE_COL).reset_index(drop=True)

    n_missing = int(df[TARGET_COL].isna().sum())
    if n_missing:
        missing_dates = df.loc[df[TARGET_COL].isna(), DATE_COL].dt.date.tolist()
        print(f"[load_full_history] Note: interpolating {n_missing} genuinely missing total_sales "
              f"value(s) for recursion-buffer continuity only: {missing_dates}")
        df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear")

    return df

HORIZON_DAYS = 30
HORIZON_CHECKPOINTS = (1, 7, 30)

FEATURE_SETS_TO_TEST: dict[str, list[str]] = {
    "baseline_15": list(BASELINE_FEATURES),
    "add_all_candidates": list(BASELINE_FEATURES) + list(ALL_CANDIDATES),
}

# (buffer, lag_days, rolling_window_or_None) - matches the exact shift/
# rolling semantics in src/features/build_features.py's add_time_features.
BUFFER_DEPENDENT_FEATURES: dict[str, tuple[str, int, int | None]] = {
    "lag_7_sales": ("sales", 7, None),
    "rolling_7_sales": ("sales", 1, 7),
    "rolling_14_sales": ("sales", 1, 14),
    "lag_1_fe": ("fe", 1, None),
    "lag_7_fe": ("fe", 7, None),
    "rolling_7_fe": ("fe", 1, 7),
}


def _lagged_value(buffer: dict, target_date: pd.Timestamp, lag_days: int, rolling_window: int | None) -> float:
    """Reproduce shift(lag_days) or shift(lag_days).rolling(window).mean()
    semantics against a date-keyed buffer that may contain a mix of real
    historical values and the model's own earlier predictions."""
    if rolling_window is None:
        ref_date = target_date - pd.Timedelta(days=lag_days)
        return buffer.get(ref_date, np.nan)

    end = target_date - pd.Timedelta(days=lag_days)
    window_dates = pd.date_range(end - pd.Timedelta(days=rolling_window - 1), end)
    vals = [buffer[d] for d in window_dates if d in buffer]
    return float(np.mean(vals)) if vals else np.nan


def _fit_catboost(train_df: pd.DataFrame, features: list[str], params: dict):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(random_state=42, verbose=False, allow_writing_files=False, **params)
    model.fit(train_df[features], train_df[TARGET_COL])
    return model


def _predict_catboost(model, feature_row: pd.Series, features: list[str]) -> float:
    X = pd.DataFrame([feature_row[features].to_dict()])[features]
    return float(model.predict(X)[0])


def _fit_prophet(train_df: pd.DataFrame, features: list[str], params: dict):
    from prophet import Prophet

    train = train_df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})[["ds", "y"] + features]
    model = Prophet(**params)
    for regressor in features:
        model.add_regressor(regressor)
    model.fit(train)
    return model


def _predict_prophet(model, feature_row: pd.Series, features: list[str], date: pd.Timestamp) -> float:
    row = {"ds": date, **{f: feature_row[f] for f in features}}
    forecast = model.predict(pd.DataFrame([row]))
    return float(forecast["yhat"].iloc[0])


FIT_FN = {"catboost": _fit_catboost, "prophet": _fit_prophet}


def simulate_recursive_forecast(
    family: str,
    feature_set_name: str,
    features: list[str],
    fold: Fold,
    df: pd.DataFrame,
    full_history: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
    use_true_actuals: bool = False,
) -> pd.DataFrame:
    """
    Fit once on the fold's training data, then predict forward day by day
    through the first `horizon_days` of the fold's test window, feeding
    each buffer-dependent feature from the model's own prior predictions
    for any referenced date that falls inside this forecast batch.

    `df` (the feature-selection dataset, every column clean) is used for
    training and for the actual feature rows being predicted. `full_history`
    (real sales/forecast_sales only, no gaps) seeds the recursion buffer -
    a gap in some unrelated feature column must never show up as a fake
    "missing sales day" in the lag/rolling lookups.

    `use_true_actuals=True` switches off the recursion: the buffer is fed
    the real actual for each day instead of the model's own prediction,
    reproducing Series 2's original "always-fresh-data" assumption on the
    exact same fold/day subset as the recursive run - the only way to
    isolate "does having real data help" from "these two experiments
    scored a different number of folds/days".
    """
    train_df = slice_train_window(df, DATE_COL, fold, train_window_days=None)
    test_df = (
        df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
        .iloc[:horizon_days]
    )

    params = MODEL_PARAMS[family]
    model = FIT_FN[family](train_df, features, params)

    history = full_history[full_history[DATE_COL] < fold.test_start].set_index(DATE_COL)
    sales_buffer: dict = history[TARGET_COL].to_dict()
    fe_buffer: dict = (history[TARGET_COL] - history["forecast_sales"]).to_dict()

    rows = []
    for horizon_idx, (_, row) in enumerate(test_df.iterrows(), start=1):
        d = row[DATE_COL]
        feature_row = row.copy()

        for feat_name, (buf_name, lag, window) in BUFFER_DEPENDENT_FEATURES.items():
            if feat_name not in features:
                continue
            buf = sales_buffer if buf_name == "sales" else fe_buffer
            feature_row[feat_name] = _lagged_value(buf, d, lag, window)

        if family == "catboost":
            pred = _predict_catboost(model, feature_row, features)
        else:
            pred = _predict_prophet(model, feature_row, features, d)

        rows.append({
            "family": family, "feature_set": feature_set_name, "fold_id": fold.fold_id,
            "horizon_day": horizon_idx, "date": d,
            "actual": row[TARGET_COL], "pred": pred,
            "abs_error": abs(row[TARGET_COL] - pred),
        })

        # Feed back either the model's OWN prediction (the genuinely
        # recursive, no-fresh-data case) or the true actual
        # (use_true_actuals=True - matches Series 2's assumption, used
        # only to build a fair same-fold/same-day comparison point).
        feedback_value = row[TARGET_COL] if use_true_actuals else pred
        sales_buffer[d] = feedback_value
        fe_buffer[d] = feedback_value - row["forecast_sales"]

    return pd.DataFrame(rows)


def _fold_has_continuous_feature_rows(fold: Fold, df: pd.DataFrame, horizon_days: int) -> bool:
    """
    A fold is usable for this experiment only if the feature-selection
    dataset `df` has an actual row for every one of the `horizon_days`
    calendar days in its test window - a fold whose horizon crosses a
    gap (some unrelated feature column missing on a real calendar day;
    see load_full_history's docstring) cannot be cleanly scored, since
    there is no clean feature row to predict from or compare against on
    the missing day(s).
    """
    expected_dates = pd.date_range(fold.test_start, periods=horizon_days, freq="D")
    available_dates = set(df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)][DATE_COL])
    return all(d in available_dates for d in expected_dates)


def run_recursive_horizon_experiment(max_folds: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    full_history = load_full_history()
    all_folds = build_folds(df, max_folds=max_folds)

    folds = [f for f in all_folds if _fold_has_continuous_feature_rows(f, df, HORIZON_DAYS)]
    skipped = [f for f in all_folds if f not in folds]
    if skipped:
        print(f"Skipping {len(skipped)} fold(s) whose {HORIZON_DAYS}-day horizon window crosses a real "
              f"data gap (see load_full_history docstring) and cannot be cleanly scored: "
              f"{[f.fold_id for f in skipped]}")

    print(f"Running recursive horizon test: {len(folds)} usable fold(s) (of {len(all_folds)}) x "
          f"{len(FEATURE_SETS_TO_TEST)} feature sets x {len(MODEL_PARAMS)} families x {HORIZON_DAYS}-day horizon")

    all_rows = []
    for family in MODEL_PARAMS:
        for feature_set_name, features in FEATURE_SETS_TO_TEST.items():
            for fold in folds:
                result = simulate_recursive_forecast(family, feature_set_name, features, fold, df, full_history)
                all_rows.append(result)
            print(f"  done: {family} / {feature_set_name}")

    results = pd.concat(all_rows, ignore_index=True)
    results_path = RESULTS_DIR / "recursive_horizon_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nDay-by-day results written to: {results_path}")

    summary_rows = []
    for family in MODEL_PARAMS:
        for feature_set_name in FEATURE_SETS_TO_TEST:
            subset = results[(results["family"] == family) & (results["feature_set"] == feature_set_name)]
            for h in HORIZON_CHECKPOINTS:
                at_h = subset[subset["horizon_day"] == h]
                if len(at_h) == 0:
                    continue
                summary_rows.append({
                    "family": family, "feature_set": feature_set_name, "horizon_day": h,
                    "n_folds": at_h["fold_id"].nunique(),
                    "MAE": at_h["abs_error"].mean(),
                })

    summary = pd.DataFrame(summary_rows).sort_values(["family", "feature_set", "horizon_day"])
    summary_path = RESULTS_DIR / "recursive_horizon_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary (MAE at day 1/7/30) written to: {summary_path}")
    print()
    print(summary.to_string(index=False))

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    run_recursive_horizon_experiment()
