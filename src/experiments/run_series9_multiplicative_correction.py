"""
Series 9 - Post-hoc multiplicative December correction.

Series 5-7 showed that telling the model about December as a *feature*
makes it worse. This series tries the other route: leave the model alone
and multiply its December predictions afterwards by how much busier each
December day historically runs than a typical day of the same weekday.

Method:
1. Out-of-fold predictions from the reference model (add_all_candidates,
   iterations=300 / learning_rate=0.03 / depth=4, expanding window).
2. A per-day-of-month ratio: each December day's sales relative to the
   full-dataset average for its weekday, pooled across both Decembers.
   Deliberately unrestricted by date - a genuinely recurring calendar
   pattern, not something to hide later years from.
3. December predictions multiplied by that ratio.

Result reproduced (MAE pooled over individual days):

    |                  | Overall (585 days) | December (31 days) |
    | Raw prediction   | 907.42             | 1,378.38           |
    | Corrected        | 1,127.31           | 5,527.95           |

Overall MAE degrades 24% and December MAE roughly fourfold: the model's
features already encode most of the December uplift, so multiplying by
the full historical uplift double-counts it. See docs/EXPERIMENT_LOG.md,
Series 9.

Outputs (reports/experiments/):
- series9_raw_predictions.csv          (reused by Series 11 and 12)
- december_ratio_lookup_pooled.json     (reused by Series 10)
- series9_corrected_predictions.csv
"""

from __future__ import annotations

import json

import pandas as pd

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, PROJECT_ROOT, REFERENCE_PARAMS, apply_december_ratio, collect_fold_predictions,
    pooled_mae, results_path,
)

DAILY_SALES_PATH = PROJECT_ROOT / "data" / "processed" / "sales" / "daily_sales_totals_master.csv"


def build_pooled_december_ratio() -> dict[int, float]:
    df = pd.read_csv(DAILY_SALES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    weekday_avg = df.groupby("day_of_week")["total_sales"].mean()

    dec = df[df["date"].dt.month == 12].copy()
    dec["day_of_month"] = dec["date"].dt.day
    dec["relative_intensity"] = dec["total_sales"] / dec["day_of_week"].map(weekday_avg)
    return dec.groupby("day_of_month")["relative_intensity"].mean().to_dict()


def run() -> dict:
    raw = collect_fold_predictions(REFERENCE_PARAMS, ADD_ALL_CANDIDATES, window=None)
    raw.to_csv(results_path("series9_raw_predictions.csv"), index=False)

    ratio = build_pooled_december_ratio()
    with open(results_path("december_ratio_lookup_pooled.json"), "w") as f:
        json.dump({str(k): v for k, v in ratio.items()}, f, indent=2)

    corrected = apply_december_ratio(raw, ratio, out_col="corrected_pred")
    corrected.to_csv(results_path("series9_corrected_predictions.csv"), index=False)

    return {
        "days": len(corrected),
        "overall_mae_raw": pooled_mae(corrected, "pred"),
        "overall_mae_corrected": pooled_mae(corrected, "corrected_pred"),
        "december_days": int((corrected["date"].dt.month == 12).sum()),
        "december_mae_raw": pooled_mae(corrected, "pred", december_only=True),
        "december_mae_corrected": pooled_mae(corrected, "corrected_pred", december_only=True),
    }


if __name__ == "__main__":
    for key, value in run().items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
