"""
Series 5 - Fixed-calendar Christmas period flag.

Hypothesis: a binary December 1st-31st flag (is_christmas_period) improves
the add_all_candidates configuration, on the assumption that the whole of
December behaves as "Christmas" for hospitality demand.

Result reproduced: MAE worsens from 907.77 to 919.33 (expanding window) -
the flag is deprecated. See docs/EXPERIMENT_LOG.md, Series 5.

Output: reports/experiments/series5_christmas_period_results.csv
"""

from __future__ import annotations

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, REFERENCE_PARAMS, leaderboard, results_path, run_catboost_trials,
)


def run():
    trials = {
        "add_all_candidates (reference)": (REFERENCE_PARAMS, ADD_ALL_CANDIDATES),
        "add_all_candidates_plus_is_christmas_period": (REFERENCE_PARAMS, ADD_ALL_CANDIDATES + ["is_christmas_period"]),
    }
    results = run_catboost_trials(trials, label_col="feature_set")
    results.to_csv(results_path("series5_christmas_period_results.csv"), index=False)
    return results


if __name__ == "__main__":
    res = run()
    print(leaderboard(res, ["feature_set", "train_window_days"]).to_string(index=False))
