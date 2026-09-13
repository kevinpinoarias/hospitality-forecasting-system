"""
Series 13 - Training directly on MAE loss.

Hypothesis: every model in this log is scored on MAE but trained on
CatBoost's default RMSE loss. Aligning the training objective with the
evaluation metric should lower MAE.

Method: the reference configuration retrained with loss_function='MAE' at
the same hyperparameters, then a dedicated 18-combination grid in case
MAE loss needs different hyperparameters to shine. Expanding window only.

Result reproduced:

    | RMSE loss, reference hyperparameters     | 907.77 |
    | MAE loss, same hyperparameters           | 940.50 |
    | MAE loss, best of dedicated grid         | 915.65 |
    |   (iterations=500, learning_rate=0.05, depth=4)   |

Rejected - RMSE loss still wins even after a dedicated search. See
docs/EXPERIMENT_LOG.md, Series 13.

Output: reports/experiments/series13_mae_loss_grid_results.csv
"""

from __future__ import annotations

from itertools import product

from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, REFERENCE_PARAMS, leaderboard, results_path, run_catboost_trials,
)

MAE_LOSS_GRID = [
    {"iterations": it, "learning_rate": lr, "depth": d, "loss_function": "MAE"}
    for it, lr, d in product([100, 300, 500], [0.03, 0.05, 0.1], [4, 6])
]


def run() -> dict:
    head_to_head = run_catboost_trials(
        {
            "RMSE loss (reference)": (REFERENCE_PARAMS, ADD_ALL_CANDIDATES),
            "MAE loss (same hyperparameters)": ({**REFERENCE_PARAMS, "loss_function": "MAE"}, ADD_ALL_CANDIDATES),
        },
        label_col="configuration",
        windows=[None],
    )

    grid = run_catboost_trials(
        {str(params): (params, ADD_ALL_CANDIDATES) for params in MAE_LOSS_GRID},
        label_col="params",
        windows=[None],
    )
    grid.to_csv(results_path("series13_mae_loss_grid_results.csv"), index=False)

    return {
        "head_to_head": leaderboard(head_to_head, ["configuration"]),
        "mae_loss_grid": leaderboard(grid, ["params"]),
    }


if __name__ == "__main__":
    out = run()
    print(out["head_to_head"].to_string(index=False))
    print()
    print(out["mae_loss_grid"].to_string(index=False))
