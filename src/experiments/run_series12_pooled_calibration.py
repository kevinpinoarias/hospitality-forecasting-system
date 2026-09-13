"""
Series 12 - Pooled calibration table (both years combined).

Series 11's ratio came from a single December. Pooling the residual ratio
from December 2024 (held out, Series 11) and December 2025 (out-of-fold,
Series 9) gives each day-of-month two observations instead of one.

Result reproduced - the two years disagree about how much correction is
needed, and not by a little:

    |                            | Raw MAE  | Pooled-corrected MAE |
    | 2024 (model never saw one) | 3,058.58 | 1,500.14 (better)    |
    | 2025 (model had seen 2024) | 1,378.38 | 2,001.76 (worse)     |

Pooling helps the blind-model year and hurts the experienced one, and a
blind model is not representative of any real deployment, which would
always have seen at least one prior December. Both years also contribute
to the pooled ratio, so these corrected figures are partly circular and
are not claimed as validation. Parked until a third, genuinely unseen
December exists. See docs/EXPERIMENT_LOG.md, Series 12.

Requires: series11_calibration_ratio_derivation.csv (Series 11),
          series9_raw_predictions.csv (Series 9)
Outputs (reports/experiments/):
- series12_pooled_calibration_table.csv
- december_calibration_pooled.json
"""

from __future__ import annotations

import json

import pandas as pd

from src.experiments.series_common import require_output, results_path


def run() -> dict:
    cal_path = require_output("series11_calibration_ratio_derivation.csv", "run_series11_calibration_ratio")
    raw_path = require_output("series9_raw_predictions.csv", "run_series9_multiplicative_correction")

    dec_2024 = pd.read_csv(cal_path)[["day_of_month", "actual", "pred"]].rename(
        columns={"actual": "actual_2024", "pred": "pred_2024"}
    )

    raw = pd.read_csv(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    dec_2025 = raw[raw["date"].dt.month == 12].copy()
    dec_2025["day_of_month"] = dec_2025["date"].dt.day
    dec_2025 = dec_2025[["day_of_month", "actual", "pred"]].rename(
        columns={"actual": "actual_2025", "pred": "pred_2025"}
    )

    pooled = dec_2024.merge(dec_2025, on="day_of_month")
    pooled["ratio_2024"] = pooled["actual_2024"] / pooled["pred_2024"]
    pooled["ratio_2025"] = pooled["actual_2025"] / pooled["pred_2025"]
    pooled["pooled_ratio"] = pooled[["ratio_2024", "ratio_2025"]].mean(axis=1)
    pooled.to_csv(results_path("series12_pooled_calibration_table.csv"), index=False)

    with open(results_path("december_calibration_pooled.json"), "w") as f:
        json.dump({str(k): v for k, v in pooled.set_index("day_of_month")["pooled_ratio"].items()}, f, indent=2)

    def mae(actual, pred):
        return float((pooled[actual] - pred).abs().mean())

    return {
        "days_matched": len(pooled),
        "mae_2024_raw": mae("actual_2024", pooled["pred_2024"]),
        "mae_2024_pooled_corrected (partly circular)": mae("actual_2024", pooled["pred_2024"] * pooled["pooled_ratio"]),
        "mae_2025_raw": mae("actual_2025", pooled["pred_2025"]),
        "mae_2025_pooled_corrected (partly circular)": mae("actual_2025", pooled["pred_2025"] * pooled["pooled_ratio"]),
    }


if __name__ == "__main__":
    for key, value in run().items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
