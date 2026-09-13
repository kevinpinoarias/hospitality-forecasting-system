"""
Series 7 - Named December phase flags (direct encoding).

Hypothesis: grouping December days into named phases (party season,
pre-Christmas peak, Christmas Eve dip, Boxing Day lull, recovery, Hogmanay
peak, New Year's Eve) - each backed by more training rows than a single
day-of-month - is more robust than Series 5's flat window or Series 6's
31-point curve.

Each phase flag is tested on its own, and all seven together.

Result reproduced: every variant worse than the reference (£916-923) -
deprecated. See docs/EXPERIMENT_LOG.md, Series 7.

Output: reports/experiments/series7_december_phases_results.csv
"""

from __future__ import annotations

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, REFERENCE_PARAMS, leaderboard, results_path, run_catboost_trials,
)
from src.features.build_features import DECEMBER_PHASES


def run():
    phase_flags = list(DECEMBER_PHASES)
    trials = {
        "add_all_candidates (reference)": (REFERENCE_PARAMS, ADD_ALL_CANDIDATES),
        "add_all_candidates_plus_all_phases": (REFERENCE_PARAMS, ADD_ALL_CANDIDATES + phase_flags),
    }
    for flag in phase_flags:
        trials[f"add_all_candidates_plus__{flag}"] = (REFERENCE_PARAMS, ADD_ALL_CANDIDATES + [flag])

    results = run_catboost_trials(trials, label_col="feature_set")
    results.to_csv(results_path("series7_december_phases_results.csv"), index=False)
    return results


if __name__ == "__main__":
    res = run()
    board = leaderboard(res, ["feature_set", "train_window_days"])
    expanding = board[board["train_window_days"] == "expanding"].copy()
    reference = expanding.loc[expanding["feature_set"].str.contains("reference"), "MAE_mean"].iloc[0]
    expanding["delta_vs_reference"] = expanding["MAE_mean"] - reference
    print(expanding.to_string(index=False))
