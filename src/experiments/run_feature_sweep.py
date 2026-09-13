"""
Feature engineering sweep: tests the current production feature set
against every already-engineered-but-unused candidate feature (see
src/experiments/feature_sets.py for the full design-of-experiments
rationale), using the two strongest model families found in the prior
algorithm/hyperparameter sweep (src/experiments/run_sweep.py) -
CatBoost (the overall winner) and Prophet (a strong, promising
alternative) - each held at its winning hyperparameter configuration so
this experiment isolates the FEATURE effect specifically.

Two phases:
1. Screening - baseline, leave-one-out, add-one, add-group, add-all
   (src.experiments.feature_sets.build_baseline_and_screening_sets).
2. Combination - for each family separately, the individually-best
   add-one candidates (by that family's own results) are combined and
   retested (src.experiments.feature_sets.build_top_k_combination_sets),
   since the best individual performers are not guaranteed to combine
   additively.

Every fold, every family, every training-window mode, every feature set
uses the identical row set: the engineered feature dataset filtered to
rows where every candidate AND baseline column is non-null, so no
comparison is ever confounded by different feature sets seeing different
amounts of data.

Outputs
-------
- reports/experiments/feature_sweep_results_full.csv
- reports/experiments/feature_sweep_leaderboard.csv
- reports/experiments/feature_sets_registry.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.experiments.adapters import DATE_COL, MODEL_ADAPTERS, TARGET_COL
from src.experiments.feature_sets import (
    ALL_CANDIDATES,
    FeatureSet,
    build_baseline_and_screening_sets,
    build_top_k_combination_sets,
)
from src.experiments.splits import Fold, generate_rolling_origin_folds, slice_train_window, slice_test_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINEERED_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "engineered_features.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "experiments"

# Held fixed at the winning configuration from the prior algorithm/
# hyperparameter sweep (see reports/experiments/sweep_leaderboard.csv) so
# this experiment isolates the feature effect, not a joint
# feature-x-hyperparameter search.
MODEL_PARAMS: dict[str, dict] = {
    "catboost": {"iterations": 300, "learning_rate": 0.03, "depth": 4},
    "prophet": {"changepoint_prior_scale": 0.01, "seasonality_prior_scale": 1.0, "seasonality_mode": "additive"},
}

TRAIN_WINDOW_OPTIONS: list[int | None] = [None, 365]
FAMILIES = list(MODEL_PARAMS)

DEFAULT_INITIAL_TRAIN_DAYS = 365
DEFAULT_TEST_DAYS = 60
DEFAULT_STEP_DAYS = 60


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the full engineered feature dataset, restricted to rows where
    every baseline AND candidate column is non-null - so every feature
    set tested in this sweep is evaluated on the exact same rows, and no
    comparison is confounded by different feature sets silently seeing
    different amounts of data.
    """
    from src.experiments.adapters import FEATURES as BASELINE_FEATURES

    df = pd.read_csv(ENGINEERED_FEATURES_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    required_cols = sorted(set(BASELINE_FEATURES) | set(ALL_CANDIDATES) | {DATE_COL, TARGET_COL})
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"engineered_features.csv is missing required column(s): {missing}")

    df = df.dropna(subset=required_cols).sort_values(DATE_COL).reset_index(drop=True)
    return df


def build_folds(df: pd.DataFrame, max_folds: int | None = None) -> list[Fold]:
    return generate_rolling_origin_folds(
        df[DATE_COL],
        initial_train_days=DEFAULT_INITIAL_TRAIN_DAYS,
        test_days=DEFAULT_TEST_DAYS,
        step_days=DEFAULT_STEP_DAYS,
        max_folds=max_folds,
    )


# ---------------------------------------------------------------------
# Running one fit
# ---------------------------------------------------------------------

def run_one_fit(
    feature_set: FeatureSet,
    family: str,
    fold: Fold,
    train_window_days: int | None,
    df: pd.DataFrame,
) -> dict:
    train_df = slice_train_window(df, DATE_COL, fold, train_window_days)
    test_df = slice_test_window(df, DATE_COL, fold)

    row = {
        "feature_set": feature_set.name,
        "category": feature_set.category,
        "family": family,
        "n_features": len(feature_set.features),
        "train_window_days": train_window_days if train_window_days is not None else "expanding",
        "fold_id": fold.fold_id,
        "train_start": train_df[DATE_COL].min().date() if len(train_df) else None,
        "train_end": fold.train_end.date(),
        "test_start": fold.test_start.date(),
        "test_end": fold.test_end.date(),
    }

    try:
        params = MODEL_PARAMS[family]
        result = MODEL_ADAPTERS[family](train_df, test_df, params, features=list(feature_set.features))
        result = result.dropna(subset=["actual", "pred"])
        if len(result) == 0:
            raise ValueError("no non-null predictions produced")

        metrics = regression_metrics(result["actual"], result["pred"])
        row.update({"status": "OK", "n": len(result), **metrics})
    except Exception as exc:  # noqa: BLE001 - one failed config must not kill the sweep
        row.update({"status": "ERROR", "n": 0, "error": f"{type(exc).__name__}: {exc}"})

    return row


def run_feature_sets(
    feature_sets: list[FeatureSet],
    families: list[str],
    folds: list[Fold],
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(feature_sets) * len(families) * len(TRAIN_WINDOW_OPTIONS) * len(folds)
    done = 0
    start = time.time()

    for feature_set in feature_sets:
        for family in families:
            for train_window_days in TRAIN_WINDOW_OPTIONS:
                for fold in folds:
                    rows.append(run_one_fit(feature_set, family, fold, train_window_days, df))
                    done += 1
        if verbose:
            print(f"[{done}/{total}] completed feature set '{feature_set.name}' "
                  f"({time.time() - start:.1f}s elapsed)")

    return pd.DataFrame(rows)


def aggregate_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "OK"].copy()
    grouped = (
        ok.groupby(["feature_set", "category", "family", "n_features", "train_window_days"])
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


def rank_add_one_candidates(leaderboard: pd.DataFrame, family: str) -> list[str]:
    """Return the add-one candidate feature names for one family, ranked
    best (lowest mean MAE, averaged across both train-window modes) first."""
    add_one = leaderboard[(leaderboard["family"] == family) & (leaderboard["category"] == "add_one")].copy()
    add_one["candidate"] = add_one["feature_set"].str.replace("add_one__", "", regex=False)
    ranked = add_one.groupby("candidate")["MAE_mean"].mean().sort_values()
    return list(ranked.index)


# ---------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------

def run_full_feature_sweep(max_folds: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    folds = build_folds(df, max_folds=max_folds)

    print(f"Loaded {len(df)} rows spanning {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print(f"Folds: {len(folds)}  |  Families: {FAMILIES}  |  Train-window options: {TRAIN_WINDOW_OPTIONS}")

    screening_sets = build_baseline_and_screening_sets()
    print(f"\nPhase 1 (screening): {len(screening_sets)} feature sets x {len(FAMILIES)} families "
          f"x {len(TRAIN_WINDOW_OPTIONS)} window modes x {len(folds)} folds = "
          f"{len(screening_sets) * len(FAMILIES) * len(TRAIN_WINDOW_OPTIONS) * len(folds)} fits")

    phase1_results = run_feature_sets(screening_sets, FAMILIES, folds, df)
    phase1_leaderboard = aggregate_leaderboard(phase1_results)

    print("\nPhase 2 (combination): building top-K combinations per family from Phase 1 add-one rankings")
    phase2_sets_by_family: dict[str, list[FeatureSet]] = {}
    phase2_results_frames = [phase1_results]

    for family in FAMILIES:
        ranked = rank_add_one_candidates(phase1_leaderboard, family)
        combo_sets = build_top_k_combination_sets(ranked, k_values=(3, 5, 8))
        # Namespace combo set names per family - the "best 3" differ per family.
        combo_sets = [
            FeatureSet(
                name=f"{family}__{s.name}",
                features=s.features,
                category=s.category,
                hypothesis=s.hypothesis,
                added=s.added,
            )
            for s in combo_sets
        ]
        phase2_sets_by_family[family] = combo_sets

        print(f"  {family}: top candidates ranked {ranked[:8]}")
        combo_results = run_feature_sets(combo_sets, [family], folds, df)
        phase2_results_frames.append(combo_results)

    all_results = pd.concat(phase2_results_frames, ignore_index=True)
    leaderboard = aggregate_leaderboard(all_results)

    results_path = RESULTS_DIR / "feature_sweep_results_full.csv"
    all_results.to_csv(results_path, index=False)
    print(f"\nFull results written to: {results_path}")

    leaderboard_path = RESULTS_DIR / "feature_sweep_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"Leaderboard written to: {leaderboard_path}")

    # Registry: every feature set actually tested, with its exact feature
    # list and hypothesis - the "what was tried and why" record.
    registry = {}
    for s in screening_sets:
        registry[s.name] = {
            "features": list(s.features), "category": s.category, "hypothesis": s.hypothesis,
            "added": list(s.added), "removed": list(s.removed),
        }
    for family, combo_sets in phase2_sets_by_family.items():
        for s in combo_sets:
            registry[s.name] = {
                "features": list(s.features), "category": s.category, "hypothesis": s.hypothesis,
                "added": list(s.added), "removed": list(s.removed), "family_specific": family,
            }

    registry_path = RESULTS_DIR / "feature_sets_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Feature set registry written to: {registry_path}")

    if len(leaderboard):
        best = leaderboard.iloc[0]
        print(f"\nBest overall: {best['feature_set']} / {best['family']} "
              f"(train window: {best['train_window_days']}) - mean MAE £{best['MAE_mean']:.2f}")

    return {"results": all_results, "leaderboard": leaderboard, "registry": registry}


if __name__ == "__main__":
    run_full_feature_sweep()
