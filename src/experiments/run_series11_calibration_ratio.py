"""
Series 11 - Multiplicative correction using a genuinely out-of-sample
calibration ratio.

Series 9 and 10 used a *raw* historical December uplift, which double-
counts what the model's own features already capture. The fix is to
calibrate against the model's residual gap instead: how far off the model
actually is on December days after its features have done their work.

Method:
1. Train on everything before 2024-12-01 and predict December 2024, which
   the model has never seen - so actual/pred is a real, non-circular
   residual ratio per day.
2. Apply that per-day ratio to Series 9's reference predictions for
   December.

Result reproduced (MAE pooled over individual days):

    |                                  | Overall  | December |
    | Raw (uncorrected)                | 907.42   | 1,378.38 |
    | Calibration-corrected            | 1,046.53 | 4,003.53 |
    | Series 9 naive ratio, for scale  | 1,127.31 | 5,527.95 |

Rejected, but directionally right: calibrating against the model's own
residual gap clearly beats applying the raw historical uplift, yet it is
still worse than no correction at all. See docs/EXPERIMENT_LOG.md,
Series 11.

Requires: series9_raw_predictions.csv (Series 9)
Outputs (reports/experiments/):
- series11_calibration_ratio_derivation.csv   (reused by Series 12)
- series11_calibration_corrected_predictions.csv
"""

from __future__ import annotations

import pandas as pd

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, DATE_COL, REFERENCE_PARAMS, TARGET_COL, apply_december_ratio,
    fit_predict_catboost_seeded, load_data, pooled_mae, require_output, results_path,
)

CALIBRATION_CUTOFF = "2024-12-01"
CALIBRATION_END = "2025-01-01"


def derive_calibration_ratio() -> tuple[pd.DataFrame, int]:
    df = load_data()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    train = df[df[DATE_COL] < CALIBRATION_CUTOFF]
    held_out_dec = df[(df[DATE_COL] >= CALIBRATION_CUTOFF) & (df[DATE_COL] < CALIBRATION_END)]

    pred = fit_predict_catboost_seeded(train, held_out_dec, REFERENCE_PARAMS, ADD_ALL_CANDIDATES, seed=42)
    cal = pd.DataFrame({
        "date": held_out_dec[DATE_COL].values,
        "actual": held_out_dec[TARGET_COL].values,
        "pred": pred,
    })
    cal["day_of_month"] = pd.to_datetime(cal["date"]).dt.day
    cal["calibration_ratio"] = cal["actual"] / cal["pred"]
    return cal, len(train)


def run() -> dict:
    raw_path = require_output("series9_raw_predictions.csv", "run_series9_multiplicative_correction")

    cal, train_rows = derive_calibration_ratio()
    cal.to_csv(results_path("series11_calibration_ratio_derivation.csv"), index=False)
    ratio = cal.set_index("day_of_month")["calibration_ratio"].to_dict()

    raw = pd.read_csv(raw_path)
    corrected = apply_december_ratio(raw, ratio, out_col="corrected_pred_calibration")
    corrected.to_csv(results_path("series11_calibration_corrected_predictions.csv"), index=False)

    return {
        "calibration_train_rows": train_rows,
        "overall_mae_raw": pooled_mae(corrected, "pred"),
        "overall_mae_corrected": pooled_mae(corrected, "corrected_pred_calibration"),
        "december_mae_raw": pooled_mae(corrected, "pred", december_only=True),
        "december_mae_corrected": pooled_mae(corrected, "corrected_pred_calibration", december_only=True),
    }


if __name__ == "__main__":
    for key, value in run().items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
