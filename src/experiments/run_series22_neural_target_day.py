"""
Series 22 - the neural networks, now allowed to see the day they are predicting.

Series 20 found V2 (38 features) makes both networks worse. But the original networks read
only the `window` days BEFORE the one they predict - never that day's own calendar, the
manager's forecast or weather, which are exactly the inputs the tree models lean on. This
series gives them that day (the window now ENDS at the predicted day; the target itself is
never an input) and re-asks the V1-vs-V2 question on the best settings Series 20 found.

  stage D   the 3 best fresh-seed-confirmed settings per network and feature set from Series 20
            stage C, re-run with the target day visible, on the SAME 5 fresh seeds (100-104) -
            so each with/without comparison is paired seed for seed
  stage E   how much history does the network still need? The best setting per network and
            feature set with the window swept over {1, 3, 7, 14, 30} days (window 1 = the predicted
            day's own features only), 3 seeds each

Run:  python -m src.experiments.run_series22_neural_target_day [--workers 10] [--pilot]
(needs Series 20's results: src.experiments.run_series20_neural_v2)
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np

from src.experiments import run_series20_neural_v2 as s20
from src.experiments.series19_21_common import (
    OUT_DIR, init_worker_cpu, load_protocol, neural_task, read_done, run_parallel, test_days,
)

RESULTS = OUT_DIR / "series22_neural_target_day.jsonl"
SEEDS_D = s20.SEEDS_C  # 100-104: the seeds the "without target day" Series 20 stage C runs used
SEEDS_E = [100, 101, 102]
WINDOWS_E = [1, 3, 7, 14, 30]


def confirmed_best(top: int = 3) -> dict[tuple[str, str], list[dict]]:
    """For each (network, feature set): the `top` stage-C settings with the lowest fresh-seed MAE
    (on the days every stage-C run predicted), best first. Each entry: {hp, mae}."""
    import pandas as pd

    done = read_done(s20.RESULTS)
    c_recs = [r for r in done.values() if r["kind"] == "C"]
    if not c_recs:
        raise SystemExit("Series 20 stage C has not been run yet")
    days = set.intersection(*(set(r["dates"]) for r in c_recs))
    actual = test_days(*load_protocol()).set_index("date")["actual"]
    rows = []
    for r in c_recs:
        s = pd.Series(r["pred"], index=r["dates"])
        s = s[s.index.isin(days)]
        rows.append({"arch": r["arch"], "fs": r["feature_set"], "hp_key": s20.hp_key(r["arch"], r["hp"]), "hp": r["hp"],
                     "mae": float(np.mean(np.abs(s.to_numpy() - actual.loc[s.index].to_numpy())))})
    df = pd.DataFrame(rows)
    out = {}
    for (arch, fs), g in df.groupby(["arch", "fs"]):
        means = g.groupby("hp_key").agg(mae=("mae", "mean"), hp=("hp", "first")).sort_values("mae").head(top)
        out[(arch, fs)] = [{"hp": row.hp, "mae": row.mae} for row in means.itertuples()]
    return out


def make(task_id: str, kind: str, arch: str, hp: dict, fs: str, seed: int) -> dict:
    task = s20.make(task_id, kind, arch, hp, fs, seed)
    task["include_target_day"] = True
    return task


def stage_d_tasks(best: dict) -> list[dict]:
    tasks = []
    for (arch, fs), settings in best.items():
        for rank, setting in enumerate(settings, start=1):
            for seed in SEEDS_D:
                tasks.append(make(f"D_{arch}_{fs}_top{rank}_{s20.hp_key(arch, setting['hp'])}_s{seed}", "D", arch, setting["hp"], fs, seed))
    return tasks


def stage_e_tasks(best: dict) -> list[dict]:
    tasks = []
    for (arch, fs), settings in best.items():
        base = settings[0]["hp"]
        for window, seed in product(WINDOWS_E, SEEDS_E):
            hp = {**base, "window": window}
            tasks.append(make(f"E_{arch}_{fs}_w{window}_s{seed}", "E", arch, hp, fs, seed))
    return tasks


def main(workers: int = 10, pilot: bool = False) -> None:
    best = confirmed_best()
    if pilot:
        tasks = stage_e_tasks(best)[:2]
        run_parallel(tasks, neural_task, RESULTS, workers, init_worker_cpu, "series22 pilot")
        return
    run_parallel(stage_d_tasks(best), neural_task, RESULTS, workers, init_worker_cpu, "series22 stage D")
    run_parallel(stage_e_tasks(best), neural_task, RESULTS, workers, init_worker_cpu, "series22 stage E")
    print(f"\nDone: {len(read_done(RESULTS))} tasks in {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    main(args.workers, args.pilot)
