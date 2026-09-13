"""
Series 16 - Seed bagging, and the discovery of seed selection bias.

Hypothesis: averaging the Series 14 log1p CatBoost over many random seeds
(bagging) reduces variance and improves on its 892.38 MAE. Same-model seed
bagging draws on training randomness, so it avoids the error-correlation
problem that sank Series 15's cross-model blend.

Method:
1. The log1p winner retrained with seeds 0-24 on every fold; predictions
   averaged at increasing bag sizes to check convergence.
2. The distribution of single-seed MAE across those 25 seeds.
3. The same 26-seed sweep (0-24, plus 42) for the raw-scale reference, to
   test whether seed 42 - the fixed random_state used in every fit since
   Series 1, purely by convention - is an unusually favourable draw.
4. A seed-noise-controlled comparison: 25-seed bagged averages for both
   configurations, with seed 42 excluded from both.

Result reproduced:

    | Bag size | 1 (seed 42) | 2      | 5      | 10     | 15     | 20     | 25     |
    | MAE      | 892.38      | 916.34 | 913.34 | 912.62 | 910.95 | 909.59 | 910.78 |

    Log1p, seeds 0-24:     min 901.33  mean 922.01  max 944.16  std 9.51
    Reference, seeds 0-24: min 909.34  mean 917.70  max 926.66  std 4.63
    Seed 42 falls below every one of the 25, for both configurations.

    | Seed-noise-controlled, 25-seed bag | Reference 914.12 | Log1p 910.78 |
    | Gap                                | 3.35 (vs. 15.39 at seed 42 alone) |

Bagging does not beat 892.38 - because 892.38 was never a representative
expectation. Seed 42 sits at the favourable tail of the seed distribution
for these folds and boosts log1p about three times as much as the
reference, so every single-seed comparison in Series 1-15 could misstate
an effect's size. Log1p still wins honestly, by £3.35 rather than £15.39.
Recommendation: the 15-25-seed bagged log1p ensemble, expected MAE ~£910.
See docs/EXPERIMENT_LOG.md, Series 16.

Requires: series14_plot_data.csv (Series 14)
Outputs (reports/experiments/):
- series16_bagging_raw_predictions.csv      (reused by Series 17)
- series16_reference_seed_predictions.csv
"""

from __future__ import annotations

import pandas as pd

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, DATE_COL, LOG1P_PARAMS, REFERENCE_PARAMS, TARGET_COL, build_folds,
    fit_predict_catboost_seeded, fold_mean_metrics, load_data, require_output, results_path,
    slice_test_window, slice_train_window,
)

LOG1P_SEEDS = list(range(25))
REFERENCE_SEEDS = list(range(25)) + [42]
BAG_SIZES = [2, 3, 5, 10, 15, 20, 25]


def seeded_fold_predictions(params: dict, seeds: list[int], log1p_target: bool) -> pd.DataFrame:
    df = load_data()
    rows = []
    for fold in build_folds(df):
        train_df = slice_train_window(df, DATE_COL, fold, None)
        test_df = slice_test_window(df, DATE_COL, fold)
        for seed in seeds:
            pred = fit_predict_catboost_seeded(
                train_df, test_df, params, ADD_ALL_CANDIDATES, seed=seed, log1p_target=log1p_target,
            )
            for d, actual, p in zip(test_df[DATE_COL], test_df[TARGET_COL], pred):
                rows.append({"date": d, "fold_id": fold.fold_id, "actual": actual, "seed": seed, "pred": p})
    return pd.DataFrame(rows)


def bagged(preds: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    sub = preds[preds["seed"].isin(seeds)]
    return sub.groupby(["date", "fold_id", "actual"], as_index=False)["pred"].mean()


def per_seed_mae(preds: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"seed": seed, "MAE": fold_mean_metrics(g, "pred")["MAE_mean"]} for seed, g in preds.groupby("seed")
    ]).sort_values("MAE").reset_index(drop=True)


def _distribution(maes: pd.Series) -> dict:
    return {"min": maes.min(), "mean": maes.mean(), "median": maes.median(), "max": maes.max(), "std": maes.std()}


def run() -> dict:
    seed_42_log1p = pd.read_csv(require_output("series14_plot_data.csv", "run_series14_log1p_target"), parse_dates=["date"])

    log_preds = seeded_fold_predictions(LOG1P_PARAMS, LOG1P_SEEDS, log1p_target=True)
    log_preds.to_csv(results_path("series16_bagging_raw_predictions.csv"), index=False)

    ref_preds = seeded_fold_predictions(REFERENCE_PARAMS, REFERENCE_SEEDS, log1p_target=False)
    ref_preds.to_csv(results_path("series16_reference_seed_predictions.csv"), index=False)

    convergence = [{"bag_size": "1 (seed=42, Series 14)", **fold_mean_metrics(seed_42_log1p, "log_pred")}]
    convergence += [
        {"bag_size": str(n), **fold_mean_metrics(bagged(log_preds, list(range(n))), "pred")} for n in BAG_SIZES
    ]

    log_seed_maes = per_seed_mae(log_preds)
    ref_seed_maes = per_seed_mae(ref_preds)
    ref_0_24 = ref_seed_maes[ref_seed_maes["seed"] != 42]

    ref_bag = fold_mean_metrics(bagged(ref_preds, LOG1P_SEEDS), "pred")["MAE_mean"]
    log_bag = fold_mean_metrics(bagged(log_preds, LOG1P_SEEDS), "pred")["MAE_mean"]

    return {
        "convergence": pd.DataFrame(convergence),
        "log1p_seed_distribution": _distribution(log_seed_maes["MAE"]),
        "reference_seed_distribution_0_24": _distribution(ref_0_24["MAE"]),
        "reference_seed_42": float(ref_seed_maes.loc[ref_seed_maes["seed"] == 42, "MAE"].iloc[0]),
        "bagged_25_reference": ref_bag,
        "bagged_25_log1p": log_bag,
        "bagged_gap": ref_bag - log_bag,
    }


if __name__ == "__main__":
    out = run()
    print(out["convergence"][["bag_size", "MAE_mean", "MAE_std"]].round(2).to_string(index=False))
    print()
    for key in ["log1p_seed_distribution", "reference_seed_distribution_0_24"]:
        print(key, {k: round(v, 2) for k, v in out[key].items()})
    print(f"reference seed 42: {out['reference_seed_42']:.2f}")
    print(f"bagged 25 seeds - reference: {out['bagged_25_reference']:.2f}  log1p: {out['bagged_25_log1p']:.2f}  gap: {out['bagged_gap']:.2f}")
