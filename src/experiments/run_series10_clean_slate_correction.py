"""
Series 10 - Multiplicative correction on a "clean slate" feature set.

Hypothesis: Series 9's correction double-counted because two explicit
holiday flags (is_christmas_break, is_school_holiday) already tell the
model about December. Removing them should leave the multiplier a clean
slate to correct against.

Method: identical to Series 9, but the base model is trained without those
two features, then the same pooled December ratio is applied.

Result reproduced (MAE pooled over individual days):

    |                         | Overall  | December |
    | Clean slate, raw        | 918.58   | 1,378.56 |
    | Clean slate, corrected  | 1,131.60 | 5,398.37 |

Removing the two flags barely moves the outcome - the December uplift is
also carried by the broader calendar features (month, day_of_year and
their cyclical encodings), which were not removed. See
docs/EXPERIMENT_LOG.md, Series 10.

Requires: december_ratio_lookup_pooled.json (Series 9)
Output:   reports/experiments/series10_clean_slate_results.csv
"""

from __future__ import annotations

import json

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, REFERENCE_PARAMS, apply_december_ratio, collect_fold_predictions, pooled_mae,
    require_output, results_path,
)

REDUNDANT_HOLIDAY_FLAGS = ["is_christmas_break", "is_school_holiday"]


def run() -> dict:
    ratio_path = require_output("december_ratio_lookup_pooled.json", "run_series9_multiplicative_correction")
    with open(ratio_path) as f:
        ratio = {int(k): v for k, v in json.load(f).items()}

    clean_slate = [f for f in ADD_ALL_CANDIDATES if f not in REDUNDANT_HOLIDAY_FLAGS]
    preds = collect_fold_predictions(REFERENCE_PARAMS, clean_slate, window=None)
    corrected = apply_december_ratio(preds, ratio, out_col="corrected_pred")
    corrected.to_csv(results_path("series10_clean_slate_results.csv"), index=False)

    return {
        "feature_count": len(clean_slate),
        "overall_mae_raw": pooled_mae(corrected, "pred"),
        "overall_mae_corrected": pooled_mae(corrected, "corrected_pred"),
        "december_mae_raw": pooled_mae(corrected, "pred", december_only=True),
        "december_mae_corrected": pooled_mae(corrected, "corrected_pred", december_only=True),
    }


if __name__ == "__main__":
    for key, value in run().items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
