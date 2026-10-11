"""
Series 21 - shallower trees, more iterations: a grid search around the final model.

The final model is a 25-seed bagged log1p CatBoost with depth 4, 300 iterations and
learning rate 0.05. This series asks whether shallower trees (depth 2-3) with many
more iterations do better. Every grid point is scored under the final model's exact
protocol (25 seeds, 10 rolling-origin folds, 585 test days), so each figure is
directly comparable with its £909.57.

  grid       depth {2, 3, 4} x iterations {300, 500, 800, 1200, 2000} x learning rate
             {0.02, 0.03, 0.05, 0.08} = 60 configurations (depth 4 / 300 / 0.05 is the
             current model)
  confirm    the 5 best configurations and the current model re-run on 25 FRESH seeds
             (100-124): the best of 60 on one set of seeds is partly seed luck, so the
             honest figure is the one from seeds that did not choose it.

Run:  python -m src.experiments.run_series21_catboost_depth_iterations [--workers 10] [--pilot]
"""

from __future__ import annotations

import argparse
from itertools import product

from src.experiments.series19_21_common import (
    FINAL_PARAMS, FINAL_SEEDS_25, OUT_DIR, catboost_task, init_worker, read_done, run_parallel,
)
from src.models.final_model import FINAL_FEATURES

RESULTS = OUT_DIR / "series21_catboost_grid.jsonl"
DEPTHS = [2, 3, 4]
ITERATIONS = [300, 500, 800, 1200, 2000]
LEARNING_RATES = [0.02, 0.03, 0.05, 0.08]
FRESH_SEEDS = list(range(100, 125))


def config_id(depth: int, iterations: int, lr: float) -> str:
    return f"d{depth}_i{iterations}_lr{lr}"


def grid_tasks() -> list[dict]:
    tasks = []
    for depth, iterations, lr in product(DEPTHS, ITERATIONS, LEARNING_RATES):
        tasks.append({
            "id": config_id(depth, iterations, lr), "kind": "grid", "features": list(FINAL_FEATURES),
            "params": {"iterations": iterations, "learning_rate": lr, "depth": depth}, "seeds": FINAL_SEEDS_25,
        })
    return tasks


def confirm_tasks(grid: dict[str, dict], top: int = 5) -> list[dict]:
    current = config_id(FINAL_PARAMS["depth"], FINAL_PARAMS["iterations"], FINAL_PARAMS["learning_rate"])
    ranked = sorted((r for r in grid.values() if r["id"] != current), key=lambda r: r["bag"]["MAE"])[:top]
    tasks = []
    for row in ranked + [grid[current]]:
        tasks.append({
            "id": f"confirm_{row['id']}", "kind": "confirm", "features": list(FINAL_FEATURES),
            "params": row["params"], "seeds": FRESH_SEEDS,
        })
    return tasks


def main(workers: int = 10, pilot: bool = False) -> None:
    tasks = grid_tasks()
    if pilot:
        tasks = [t for t in tasks if t["id"] in {"d4_i300_lr0.05", "d2_i500_lr0.03"}]
    grid = run_parallel(tasks, catboost_task, RESULTS, workers, init_worker, "series21 grid")
    if pilot:
        return
    run_parallel(confirm_tasks(grid), catboost_task, RESULTS, workers, init_worker, "series21 confirm")
    done = read_done(RESULTS)
    best = min((r for r in done.values() if r["kind"] == "grid"), key=lambda r: r["bag"]["MAE"])
    print(f"\nBest grid point: {best['id']} - pooled MAE £{best['bag']['MAE']:.2f} "
          f"(current model {done['d4_i300_lr0.05']['bag']['MAE']:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    main(args.workers, args.pilot)
