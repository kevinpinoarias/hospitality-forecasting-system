"""
Systematic model-comparison sweep: rolling-origin backtesting x full
hyperparameter grid search x expanding/sliding training-window options,
across every algorithm family in src/experiments/adapters.py.

This does not replace the production training scripts
(src/models/train_xgboost.py etc.) - it is a separate, offline research
tool for finding which family/config/training-window combination
generalises best across the ~2.5 years of data now available, before
anyone decides to promote a new configuration into production.

Usage
-----
Report the fold plan and a fit-count estimate without running anything:
    python -m src.experiments.run_sweep --plan-only

Run a small, fast pilot (few folds, first couple of grid points per
family, expanding window only) to sanity-check the harness and measure
real per-family fit timing, extrapolated to a full-sweep estimate:
    python -m src.experiments.run_sweep --pilot

Run the full configured sweep (every family's full grid x every
TRAIN_WINDOW_OPTIONS value x every rolling-origin fold):
    python -m src.experiments.run_sweep --full

Outputs (--full only)
----------------------
- reports/experiments/sweep_results_full.csv    (one row per fit)
- reports/experiments/sweep_leaderboard.csv     (aggregated across folds)
- reports/experiments/sweep_summary.md
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.experiments.adapters import DATE_COL, MODEL_ADAPTERS, SARIMAX_EXOG_FEATURES, TARGET_COL
from src.experiments.grids import MODEL_GRIDS, TRAIN_WINDOW_OPTIONS, total_planned_fits
from src.experiments.splits import Fold, generate_rolling_origin_folds, slice_test_window, slice_train_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "model_features.csv"
ENGINEERED_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "engineered_features.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "experiments"

DEFAULT_INITIAL_TRAIN_DAYS = 365
DEFAULT_TEST_DAYS = 60
DEFAULT_STEP_DAYS = 60

# Families sourced from engineered_features.csv (needs columns beyond the
# 15 final model features, e.g. lag_7_sales) rather than model_features.csv.
ENGINEERED_DATA_FAMILIES = {"sarimax"}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both feature datasets, restricted to the date range they share -
    every fold is defined against this shared range so every algorithm
    family is scored on exactly the same test periods.
    """
    model_df = pd.read_csv(MODEL_FEATURES_PATH)
    model_df[DATE_COL] = pd.to_datetime(model_df[DATE_COL])
    model_df = model_df.sort_values(DATE_COL).reset_index(drop=True)

    eng_df = pd.read_csv(ENGINEERED_FEATURES_PATH)
    eng_df[DATE_COL] = pd.to_datetime(eng_df[DATE_COL])
    eng_df = eng_df.sort_values(DATE_COL).reset_index(drop=True)

    shared_min, shared_max = model_df[DATE_COL].min(), model_df[DATE_COL].max()
    eng_df = eng_df[(eng_df[DATE_COL] >= shared_min) & (eng_df[DATE_COL] <= shared_max)].reset_index(drop=True)

    # A handful of NaNs at the very start of the series (e.g. lag_7_sales's
    # first 7 rows) are expected and harmless - every fold's earliest
    # training data starts well after this, and fit_predict_sarimax
    # dropna()s its own slice defensively regardless. Only warn.
    for col in SARIMAX_EXOG_FEATURES:
        n_missing = int(eng_df[col].isna().sum())
        if n_missing:
            print(f"[load_data] Note: {n_missing} row(s) missing '{col}' near the start of the series "
                  "(expected, handled per-fold).")

    return model_df, eng_df


# ---------------------------------------------------------------------
# Sweep plan
# ---------------------------------------------------------------------

def build_folds(
    model_df: pd.DataFrame,
    initial_train_days: int,
    test_days: int,
    step_days: int,
    max_folds: int | None = None,
) -> list[Fold]:
    return generate_rolling_origin_folds(
        model_df[DATE_COL],
        initial_train_days=initial_train_days,
        test_days=test_days,
        step_days=step_days,
        max_folds=max_folds,
    )


def print_plan(folds: list[Fold], model_names: list[str]) -> None:
    print(f"Planned folds: {len(folds)}")
    if folds:
        print(f"  Fold 0:  train {folds[0].train_start.date()} -> {folds[0].train_end.date()} "
              f"(exclusive), test {folds[0].test_start.date()} -> {folds[0].test_end.date()} (exclusive)")
        print(f"  Fold {len(folds) - 1}: train {folds[-1].train_start.date()} -> {folds[-1].train_end.date()} "
              f"(exclusive), test {folds[-1].test_start.date()} -> {folds[-1].test_end.date()} (exclusive)")
    print(f"Train-window options: {TRAIN_WINDOW_OPTIONS}")

    fits = total_planned_fits(len(folds), model_names)
    print("\nPlanned fits per family (grid_size x train_window_options x folds):")
    for name, n in fits.items():
        print(f"  {name:16s} {len(MODEL_GRIDS[name]):4d} params  ->  {n:6d} fits")
    print(f"  {'TOTAL':16s} {'':4s}         ->  {sum(fits.values()):6d} fits")


# ---------------------------------------------------------------------
# Running one fit
# ---------------------------------------------------------------------

def run_one_fit(
    family: str,
    params: dict,
    fold: Fold,
    train_window_days: int | None,
    model_df: pd.DataFrame,
    eng_df: pd.DataFrame,
) -> dict:
    df = eng_df if family in ENGINEERED_DATA_FAMILIES else model_df
    train_df = slice_train_window(df, DATE_COL, fold, train_window_days)
    test_df = slice_test_window(df, DATE_COL, fold)

    # Record the actual training data actually used, not the fold's nominal
    # (always dataset-start) train_start - those differ whenever a sliding
    # train_window_days is applied, and the log must reflect what a model
    # really trained on, not the unrestricted fold boundary.
    row = {
        "family": family,
        "params": str(params),
        "train_window_days": train_window_days if train_window_days is not None else "expanding",
        "fold_id": fold.fold_id,
        "train_start": train_df[DATE_COL].min().date() if len(train_df) else None,
        "train_end": fold.train_end.date(),
        "test_start": fold.test_start.date(),
        "test_end": fold.test_end.date(),
    }

    try:
        result = MODEL_ADAPTERS[family](train_df, test_df, params)
        result = result.dropna(subset=["actual", "pred"])
        if len(result) == 0:
            raise ValueError("no non-null predictions produced")

        metrics = regression_metrics(result["actual"], result["pred"])
        row.update({"status": "OK", "n": len(result), **metrics})
    except Exception as exc:  # noqa: BLE001 - one failed config must not kill the sweep
        row.update({"status": "ERROR", "n": 0, "error": f"{type(exc).__name__}: {exc}"})

    return row


# ---------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------

def run_sweep_configs(
    model_names: list[str],
    folds: list[Fold],
    train_window_options: list[int | None],
    grids: dict[str, list[dict]],
    model_df: pd.DataFrame,
    eng_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    total = sum(len(grids[name]) * len(train_window_options) * len(folds) for name in model_names)
    done = 0
    start = time.time()

    for family in model_names:
        family_start = time.time()
        for params in grids[family]:
            for train_window_days in train_window_options:
                for fold in folds:
                    rows.append(run_one_fit(family, params, fold, train_window_days, model_df, eng_df))
                    done += 1
        if verbose:
            elapsed = time.time() - family_start
            n_family_fits = len(grids[family]) * len(train_window_options) * len(folds)
            print(f"[{done}/{total}] {family}: {n_family_fits} fits in {elapsed:.1f}s "
                  f"({elapsed / max(n_family_fits, 1):.2f}s/fit)")

    if verbose:
        print(f"\nSweep finished in {time.time() - start:.1f}s")

    return pd.DataFrame(rows)


def aggregate_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "OK"].copy()
    grouped = (
        ok.groupby(["family", "params", "train_window_days"])
        .agg(
            n_folds=("fold_id", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAPE_mean=("MAPE", "mean"),
            Bias_mean=("Bias", "mean"),
        )
        .reset_index()
        .sort_values("MAE_mean")
        .reset_index(drop=True)
    )
    return grouped


def write_summary_markdown(leaderboard: pd.DataFrame, results: pd.DataFrame, output_path: Path) -> None:
    n_errors = int((results["status"] == "ERROR").sum())
    best_overall = leaderboard.iloc[0] if len(leaderboard) else None
    best_per_family = (
        leaderboard.sort_values("MAE_mean").groupby("family", as_index=False).first().sort_values("MAE_mean")
    )

    window_effect = (
        leaderboard.groupby("train_window_days")["MAE_mean"].mean().reset_index()
        .rename(columns={"MAE_mean": "avg_MAE_across_all_configs"})
    )

    lines = [
        "# Systematic Sweep Summary",
        "",
        f"{len(results)} total fits attempted, {n_errors} failed (see sweep_results_full.csv for details).",
        "",
        "## Best config overall",
        "",
    ]
    if best_overall is not None:
        lines += [
            f"**{best_overall['family']}** with `{best_overall['params']}` "
            f"(train window: {best_overall['train_window_days']}) - "
            f"mean MAE £{best_overall['MAE_mean']:.2f} across {int(best_overall['n_folds'])} folds.",
            "",
        ]

    lines += ["## Best config per family", "", best_per_family.to_markdown(index=False), ""]
    lines += [
        "## Expanding vs. sliding training window - average effect across all configs",
        "",
        "Directly tests this project's own open question (see README's \"Validation Strategy "
        "and Generalisation\"): does more historical data actually help on average, or does an "
        "older, possibly-stale period hurt more than it contributes?",
        "",
        window_effect.to_markdown(index=False),
        "",
        "## Full leaderboard (top 30 by mean MAE)",
        "",
        leaderboard.head(30).to_markdown(index=False),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Pilot mode
# ---------------------------------------------------------------------

def run_pilot(model_names: list[str] | None = None, max_folds: int = 2, combos_per_family: int = 2) -> None:
    """
    Fast sanity-check run: a couple of folds, the first `combos_per_family`
    grid points per family, expanding window only. Prints real measured
    per-family fit timing and extrapolates it to the full configured
    sweep's total estimated runtime - nothing is written to disk.
    """
    model_df, eng_df = load_data()
    names = model_names or list(MODEL_ADAPTERS)
    folds = build_folds(model_df, DEFAULT_INITIAL_TRAIN_DAYS, DEFAULT_TEST_DAYS, DEFAULT_STEP_DAYS, max_folds=max_folds)

    pilot_grids = {name: MODEL_GRIDS[name][:combos_per_family] for name in names}

    print(f"PILOT: {len(folds)} fold(s), up to {combos_per_family} param combo(s)/family, expanding window only.\n")
    results = run_sweep_configs(names, folds, [None], pilot_grids, model_df, eng_df, verbose=True)

    n_errors = int((results["status"] == "ERROR").sum())
    if n_errors:
        print(f"\n{n_errors} pilot fit(s) failed:")
        for _, row in results[results["status"] == "ERROR"].iterrows():
            print(f"  {row['family']} {row['params']} fold {row['fold_id']}: {row['error']}")

    # Extrapolate real per-fit timing (from this pilot) to the full configured sweep.
    full_folds = build_folds(model_df, DEFAULT_INITIAL_TRAIN_DAYS, DEFAULT_TEST_DAYS, DEFAULT_STEP_DAYS)
    full_fits = total_planned_fits(len(full_folds), names)

    print("\n--- Full-sweep estimate, extrapolated from this pilot's measured timing ---")
    total_est_seconds = 0.0
    for name in names:
        family_pilot_rows = results[results["family"] == name]
        n_pilot_fits = len(family_pilot_rows)
        if n_pilot_fits == 0:
            continue
        # Pilot timing isn't captured per-row here; re-derive from the
        # printed per-family elapsed time is not available post-hoc, so
        # time the family in isolation for an accurate per-fit figure.
        per_fit_seconds = _time_single_fit(name, pilot_grids[name][0], folds[0], model_df, eng_df)
        est_seconds = per_fit_seconds * full_fits[name]
        total_est_seconds += est_seconds
        print(f"  {name:16s} ~{per_fit_seconds:5.2f}s/fit x {full_fits[name]:5d} fits "
              f"~= {est_seconds / 60:6.1f} min")

    print(f"\n  {'TOTAL':16s} {'':17s}    ~= {total_est_seconds / 60:6.1f} min "
          f"({total_est_seconds / 3600:.2f} hours)")
    print("\nRun with --full to execute the entire sweep (writes results to reports/experiments/).")


def _time_single_fit(family: str, params: dict, fold: Fold, model_df: pd.DataFrame, eng_df: pd.DataFrame) -> float:
    start = time.time()
    run_one_fit(family, params, fold, None, model_df, eng_df)
    return time.time() - start


# ---------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------

def run_full_sweep(
    model_names: list[str] | None = None,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_df, eng_df = load_data()
    names = model_names or list(MODEL_ADAPTERS)
    folds = build_folds(model_df, initial_train_days, test_days, step_days, max_folds=max_folds)

    print_plan(folds, names)
    print()

    results = run_sweep_configs(names, folds, TRAIN_WINDOW_OPTIONS, MODEL_GRIDS, model_df, eng_df, verbose=True)
    results_path = RESULTS_DIR / "sweep_results_full.csv"
    results.to_csv(results_path, index=False)
    print(f"\nFull results written to: {results_path}")

    leaderboard = aggregate_leaderboard(results)
    leaderboard_path = RESULTS_DIR / "sweep_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"Leaderboard written to: {leaderboard_path}")

    summary_path = RESULTS_DIR / "sweep_summary.md"
    write_summary_markdown(leaderboard, results, summary_path)
    print(f"Summary written to: {summary_path}")

    if len(leaderboard):
        best = leaderboard.iloc[0]
        print(f"\nBest config: {best['family']} {best['params']} "
              f"(train window: {best['train_window_days']}) - mean MAE £{best['MAE_mean']:.2f}")

    return results, leaderboard


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="Print the fold plan and fit-count estimate, run nothing.")
    mode.add_argument("--pilot", action="store_true", help="Run a fast sanity-check subset and estimate full-sweep runtime.")
    mode.add_argument("--full", action="store_true", help="Run the entire configured sweep.")

    parser.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--families", nargs="*", default=None, help="Subset of model families to run (default: all).")
    args = parser.parse_args()

    names = args.families or list(MODEL_ADAPTERS)

    if args.plan_only:
        model_df, _ = load_data()
        folds = build_folds(model_df, args.initial_train_days, args.test_days, args.step_days, args.max_folds)
        print_plan(folds, names)
    elif args.pilot:
        run_pilot(model_names=names, max_folds=args.max_folds or 2)
    else:
        run_full_sweep(
            model_names=names,
            initial_train_days=args.initial_train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            max_folds=args.max_folds,
        )


if __name__ == "__main__":
    main()
