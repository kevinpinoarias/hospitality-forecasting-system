"""
Series 15 - CatBoost + Prophet ensemble.

Hypothesis: Series 1 found Prophet competitive with the lowest
fold-to-fold variance of anything tested - the classic sign of a model
making different errors from CatBoost. Blending the two should beat
either alone.

Method:
1. Prophet (Series 1 winning parameters) with all 38 features as
   regressors, on the same 10 folds.
2. Blends of the Series 14 log1p CatBoost and Prophet at every weight
   from 0 to 1 in steps of 0.05.
3. An honest check of that weight: tune it on one half of the folds and
   evaluate on the other half, both ways round.

Result reproduced:

    | CatBoost log1p (reference)                      | 892.38 |
    | Prophet, standalone                             | 954.23 |
    | 50/50 blend                                     | 903.90 |
    | Best weight on all folds (90% CatBoost)         | 891.19 |

    | Weight tuned on | Chosen | Evaluated on | Blend  | Pure CatBoost |
    | Folds 0-4       | 0.80   | Folds 5-9    | 876.17 | 871.44        |
    | Folds 5-9       | 1.00   | Folds 0-4    | 912.38 | 912.38        |

Rejected. The £1.19 gain vanishes out of sample - the weight was
overfitted to the evaluation folds - and the reason is visible directly:
CatBoost and Prophet residuals correlate at 0.892, so they make largely
the same mistakes. See docs/EXPERIMENT_LOG.md, Series 15.

Requires: series14_plot_data.csv (Series 14)
Outputs (reports/experiments/):
- series15_prophet_predictions.csv
- series15_ensemble_data.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, DATE_COL, TARGET_COL, build_folds, fold_mean_metrics, load_data, require_output,
    results_path, slice_test_window, slice_train_window,
)

PROPHET_PARAMS = {"changepoint_prior_scale": 0.01, "seasonality_prior_scale": 1.0, "seasonality_mode": "additive"}
BLEND_WEIGHTS = np.arange(0.0, 1.01, 0.05)


def prophet_fold_predictions() -> pd.DataFrame:
    from prophet import Prophet

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    df = load_data()
    records = []
    for fold in build_folds(df):
        train_df = slice_train_window(df, DATE_COL, fold, None)
        test_df = slice_test_window(df, DATE_COL, fold)
        train = train_df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})[["ds", "y"] + ADD_ALL_CANDIDATES]
        test = test_df.rename(columns={DATE_COL: "ds"})[["ds"] + ADD_ALL_CANDIDATES]

        model = Prophet(**PROPHET_PARAMS)
        for regressor in ADD_ALL_CANDIDATES:
            model.add_regressor(regressor)
        model.fit(train)
        forecast = model.predict(test)

        for d, actual, p in zip(test_df[DATE_COL], test_df[TARGET_COL], forecast["yhat"]):
            records.append({"date": d, "fold_id": fold.fold_id, "actual": actual, "prophet_pred": p})
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def _held_out_weight_check(merged: pd.DataFrame, tune_folds, eval_folds) -> dict:
    tune = merged[merged["fold_id"].isin(tune_folds)]
    evaluate = merged[merged["fold_id"].isin(eval_folds)]

    best_w, best_mae = None, np.inf
    for w in BLEND_WEIGHTS:
        mae = np.mean(np.abs(tune["actual"] - (w * tune["log_pred"] + (1 - w) * tune["prophet_pred"])))
        if mae < best_mae:
            best_w, best_mae = w, mae

    blend = best_w * evaluate["log_pred"] + (1 - best_w) * evaluate["prophet_pred"]
    return {
        "weight_chosen": round(float(best_w), 2),
        "blend_mae_held_out": float(np.mean(np.abs(evaluate["actual"] - blend))),
        "pure_catboost_mae_held_out": float(np.mean(np.abs(evaluate["actual"] - evaluate["log_pred"]))),
    }


def run() -> dict:
    catboost = pd.read_csv(require_output("series14_plot_data.csv", "run_series14_log1p_target"), parse_dates=["date"])

    prophet = prophet_fold_predictions()
    prophet.to_csv(results_path("series15_prophet_predictions.csv"), index=False)

    merged = catboost.merge(prophet[["date", "prophet_pred"]], on="date", how="inner")
    assert len(merged) == len(catboost) == len(prophet), "CatBoost and Prophet must cover the same test days"
    merged["ens_log1p_prophet_50_50"] = 0.5 * merged["log_pred"] + 0.5 * merged["prophet_pred"]
    merged["ens_ref_prophet_50_50"] = 0.5 * merged["ref_pred"] + 0.5 * merged["prophet_pred"]
    merged.to_csv(results_path("series15_ensemble_data.csv"), index=False)

    models = pd.DataFrame([
        {"model": name, **fold_mean_metrics(merged, col)}
        for name, col in [
            ("CatBoost reference (raw-scale)", "ref_pred"),
            ("CatBoost log1p (Series 14 winner)", "log_pred"),
            ("Prophet (add_all_candidates)", "prophet_pred"),
            ("Ensemble: 50/50 log1p + Prophet", "ens_log1p_prophet_50_50"),
            ("Ensemble: 50/50 reference + Prophet", "ens_ref_prophet_50_50"),
        ]
    ])

    weights = []
    for w in BLEND_WEIGHTS:
        merged["_blend"] = w * merged["log_pred"] + (1 - w) * merged["prophet_pred"]
        weights.append({"weight_on_catboost_log1p": round(float(w), 2), **fold_mean_metrics(merged, "_blend")})
    merged = merged.drop(columns="_blend")

    return {
        "models": models,
        "weight_sweep": pd.DataFrame(weights),
        "residual_correlation": float(np.corrcoef(merged["actual"] - merged["log_pred"], merged["actual"] - merged["prophet_pred"])[0, 1]),
        "held_out_tuned_0_4": _held_out_weight_check(merged, range(0, 5), range(5, 10)),
        "held_out_tuned_5_9": _held_out_weight_check(merged, range(5, 10), range(0, 5)),
    }


if __name__ == "__main__":
    out = run()
    print(out["models"].round(2).to_string(index=False))
    print()
    print(out["weight_sweep"][["weight_on_catboost_log1p", "MAE_mean", "MAE_std"]].round(2).to_string(index=False))
    print()
    print(f"residual correlation: {out['residual_correlation']:.3f}")
    print(f"tuned on folds 0-4: {out['held_out_tuned_0_4']}")
    print(f"tuned on folds 5-9: {out['held_out_tuned_5_9']}")
