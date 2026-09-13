"""
Series 8 - Hyperparameter re-tune on add_all_candidates.

Hypothesis: Series 4-7's consistent failure to improve on add_all_candidates
reflects CatBoost hyperparameters tuned in Series 1 for the smaller
15-feature space, not genuine redundancy - so re-running the full CatBoost
grid on the 38-feature set finds a better configuration.

Result reproduced: the original parameters (iterations=300,
learning_rate=0.03, depth=4) remain best at 907.77. See
docs/EXPERIMENT_LOG.md, Series 8.

Output: reports/experiments/series8_hyperparameter_retune_results.csv
"""

from __future__ import annotations

from src.experiments.grids import CATBOOST_GRID
from src.experiments.series_common import ADD_ALL_CANDIDATES, leaderboard, results_path, run_catboost_trials


def run():
    trials = {str(params): (params, ADD_ALL_CANDIDATES) for params in CATBOOST_GRID}
    results = run_catboost_trials(trials, label_col="params")
    results.to_csv(results_path("series8_hyperparameter_retune_results.csv"), index=False)
    return results


if __name__ == "__main__":
    res = run()
    print(leaderboard(res, ["params", "train_window_days"]).head(15).to_string(index=False))
