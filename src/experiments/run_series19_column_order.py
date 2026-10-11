"""
Series 19 - does the ORDER of the 38 columns change the final model's accuracy?

Column order changes CatBoost's seeded fits (documented in
src/models/final_model.py), so an unlucky or lucky order could move the headline
figure for no real reason. This series measures how much.

Every task is the final model's exact protocol (25-seed log1p CatBoost bag,
10 rolling-origin folds, 585 test days) with only the column order changed:

  baseline_order   the current order (reproduces the final model's £909.57)
  random_NNN       100 random permutations of the 38 columns (seeded, distinct)
  reversed, alphabetical, candidates_first   three structured orders, for reference
  seedset_KK       CONTROL: the current order, 30 different sets of 25 seeds -
                   how much the figure moves from seed luck alone, to compare the
                   order effect against
  confirm_*        the best 3 random orders re-run on 5 fresh seed sets, next to
                   the current order on the same sets (guards against picking the
                   luckiest of 100)

Run:  python -m src.experiments.run_series19_column_order [--workers 10] [--pilot]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.experiments.series19_21_common import (
    FINAL_PARAMS, FINAL_SEEDS_25, OUT_DIR, catboost_task, init_worker, read_done, run_parallel,
)
from src.models.final_model import FINAL_FEATURES

RESULTS = OUT_DIR / "series19_column_order.jsonl"
ORDERS = OUT_DIR / "series19_orders.json"
N_RANDOM = 100
N_SEEDSETS = 30
CONFIRM_SEEDSETS = [101, 102, 103, 104, 105]
ORDER_RNG_SEED = 19


def seed_set(k: int) -> list[int]:
    return list(range(100 * k, 100 * k + 25))


def random_orders(n: int, rng_seed: int = ORDER_RNG_SEED) -> list[list[str]]:
    """n distinct random permutations of the 38 columns, none equal to the current order."""
    rng = np.random.default_rng(rng_seed)
    seen = {tuple(FINAL_FEATURES)}
    orders: list[list[str]] = []
    while len(orders) < n:
        perm = tuple(str(x) for x in rng.permutation(FINAL_FEATURES))
        if perm not in seen:
            seen.add(perm)
            orders.append(list(perm))
    return orders


def structured_orders() -> dict[str, list[str]]:
    return {
        "reversed": list(reversed(FINAL_FEATURES)),
        "alphabetical": sorted(FINAL_FEATURES),
        "candidates_first": FINAL_FEATURES[15:] + FINAL_FEATURES[:15],
    }


def task(task_id: str, kind: str, features: list[str], seeds: list[int]) -> dict:
    return {"id": task_id, "kind": kind, "features": features, "params": FINAL_PARAMS, "seeds": seeds}


def stage1_tasks(n_random: int = N_RANDOM, n_seedsets: int = N_SEEDSETS) -> list[dict]:
    tasks = [task("baseline_order", "baseline", list(FINAL_FEATURES), FINAL_SEEDS_25)]
    for i, order in enumerate(random_orders(n_random), start=1):
        tasks.append(task(f"random_{i:03d}", "random_order", order, FINAL_SEEDS_25))
    for name, order in structured_orders().items():
        tasks.append(task(name, "structured_order", order, FINAL_SEEDS_25))
    for k in range(1, n_seedsets + 1):
        tasks.append(task(f"seedset_{k:02d}", "seed_control", list(FINAL_FEATURES), seed_set(k)))
    return tasks


def confirm_tasks(stage1: dict[str, dict], orders: dict[str, list[str]], top: int = 3) -> list[dict]:
    random_rows = sorted((r for r in stage1.values() if r["kind"] == "random_order"), key=lambda r: r["bag"]["MAE"])
    tasks = []
    for rank, row in enumerate(random_rows[:top], start=1):
        for k in CONFIRM_SEEDSETS:
            tasks.append(task(f"confirm_best{rank}_{row['id']}_s{k}", "confirm_order", orders[row["id"]], seed_set(k)))
    for k in CONFIRM_SEEDSETS:
        tasks.append(task(f"confirm_baseline_s{k}", "confirm_baseline", list(FINAL_FEATURES), seed_set(k)))
    return tasks


def main(workers: int = 10, pilot: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = stage1_tasks()
    if pilot:
        tasks = [tasks[0], tasks[1]]
    orders = {t["id"]: t["features"] for t in tasks}
    ORDERS.write_text(json.dumps(orders), encoding="utf-8")
    stage1 = run_parallel(tasks, catboost_task, RESULTS, workers, init_worker, "series19 stage 1")
    if pilot:
        return
    confirm = confirm_tasks(stage1, orders)
    orders.update({t["id"]: t["features"] for t in confirm})
    ORDERS.write_text(json.dumps(orders), encoding="utf-8")
    run_parallel(confirm, catboost_task, RESULTS, workers, init_worker, "series19 confirm")
    base = read_done(RESULTS)["baseline_order"]["bag"]
    print(f"\nBaseline order: pooled MAE £{base['MAE']:.2f} (the final model's figure is £909.57)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot", action="store_true", help="run just the baseline and one random order")
    args = parser.parse_args()
    main(args.workers, args.pilot)
