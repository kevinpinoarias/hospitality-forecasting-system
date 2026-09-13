"""
Series 6 - Empirical graded December intensity index.

Hypothesis: a walk-forward-safe, graded December signal
(december_intensity_index) that follows the real twin-peak shape of
December demand succeeds where Series 5's flat binary flag did not.

Result reproduced: still worse than the reference, more narrowly than the
binary flag - deprecated. See docs/EXPERIMENT_LOG.md, Series 6.

Output: reports/experiments/series6_december_intensity_results.csv
"""

from __future__ import annotations

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, REFERENCE_PARAMS, leaderboard, results_path, run_catboost_trials,
)


def run():
    trials = {
        "add_all_candidates (reference)": (REFERENCE_PARAMS, ADD_ALL_CANDIDATES),
        "add_all_candidates_plus_december_intensity_index": (
            REFERENCE_PARAMS, ADD_ALL_CANDIDATES + ["december_intensity_index"],
        ),
    }
    results = run_catboost_trials(trials, label_col="feature_set")
    results.to_csv(results_path("series6_december_intensity_results.csv"), index=False)
    return results


if __name__ == "__main__":
    res = run()
    print(leaderboard(res, ["feature_set", "train_window_days"]).to_string(index=False))
