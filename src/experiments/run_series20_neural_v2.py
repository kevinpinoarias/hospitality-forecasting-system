"""
Series 20 - the neural networks on the 38-feature set (V2) instead of the original 15 (V1).

The LSTM and Transformer were only ever compared with CatBoost at V1. This series runs
the repo's own networks (src/models/train_lstm.py, train_transformer.py) under the final
model's protocol - 10 rolling-origin folds, expanding window, same 950 rows - and compares
them with the final model.

  stage A   the current configurations (the production defaults, and the Series 1 winners)
            on V1, V2 and, as a diagnostic, V2 without `year`, each over 5 seeds
  stage B   a grid search over hyperparameters, on V1 and on V2, seed 42 (as Series 1):
            LSTM hidden {16,32,64} x layers {1,2} x window {7,14,30} x epochs {40,100};
            Transformer d_model {16,32,64} x heads {2,4} x window {7,14,30} x epochs {40,100}
  stage C   the 3 best grid points per network and feature set, re-run on 5 FRESH seeds -
            the best of 36 on one seed is partly luck, so the honest figure is the one
            from seeds that did not choose it

Note: like the originals, the networks read only the `window` days BEFORE the day they
predict, not that day's own calendar, forecast or weather features, and skip windows that
span a missing calendar day.

Run:  python -m src.experiments.run_series20_neural_v2 [--workers 10] [--pilot]
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np

from src.experiments.series19_21_common import (
    OUT_DIR, init_worker_cpu, load_protocol, neural_task, read_done, run_parallel, test_days,
)
from src.models.final_model import FINAL_FEATURES

RESULTS = OUT_DIR / "series20_neural.jsonl"
FEATURE_SETS = {
    "v1": FINAL_FEATURES[:15],
    "v2": list(FINAL_FEATURES),
    "v2_no_year": [f for f in FINAL_FEATURES if f != "year"],  # diagnostic: `year` is out of range in every test block
}
SEEDS_A = [42, 43, 44, 45, 46]
SEEDS_C = [100, 101, 102, 103, 104]

# Stage A: "current" networks. Production = the defaults in train_lstm.py / train_transformer.py
# (100 epochs, 30-day window); Series 1 = the winners of the log's grid (40 epochs).
CURRENT_CONFIGS = {
    "lstm_production": ("lstm", {"hidden_size": 32, "num_layers": 1, "window": 30, "epochs": 100}),
    "lstm_series1": ("lstm", {"hidden_size": 32, "num_layers": 1, "window": 14, "epochs": 40}),
    "transformer_production": ("transformer", {"d_model": 32, "nhead": 4, "window": 30, "epochs": 100}),
    "transformer_series1": ("transformer", {"d_model": 32, "nhead": 2, "window": 30, "epochs": 40}),
}


def hp_key(arch: str, hp: dict) -> str:
    if arch == "lstm":
        return f"h{hp['hidden_size']}_l{hp['num_layers']}_w{hp['window']}_e{hp['epochs']}"
    return f"d{hp['d_model']}_n{hp['nhead']}_w{hp['window']}_e{hp['epochs']}"


def make(task_id: str, kind: str, arch: str, hp: dict, feature_set: str, seed: int) -> dict:
    return {"id": task_id, "kind": kind, "arch": arch, "hp": hp, "feature_set": feature_set,
            "features": FEATURE_SETS[feature_set], "seed": seed}


def stage_a_tasks() -> list[dict]:
    return [
        make(f"A_{name}_{fs}_s{seed}", "A", arch, hp, fs, seed)
        for (name, (arch, hp)), fs, seed in product(CURRENT_CONFIGS.items(), FEATURE_SETS, SEEDS_A)
    ]


def grid_points() -> list[tuple[str, dict]]:
    points = [("lstm", {"hidden_size": h, "num_layers": n, "window": w, "epochs": e})
              for h, n, w, e in product([16, 32, 64], [1, 2], [7, 14, 30], [40, 100])]
    points += [("transformer", {"d_model": d, "nhead": n, "window": w, "epochs": e})
               for d, n, w, e in product([16, 32, 64], [2, 4], [7, 14, 30], [40, 100])]
    return points


def stage_b_tasks() -> list[dict]:
    return [make(f"B_{arch}_{hp_key(arch, hp)}_{fs}", "B", arch, hp, fs, 42)
            for (arch, hp), fs in product(grid_points(), ["v1", "v2"])]


def common_days_mae(records: list[dict], days: set[str]) -> dict[str, float]:
    """Pooled MAE of each record over the days EVERY record predicted (windows of different
    lengths skip different days, so raw MAEs would not be on the same days)."""
    import pandas as pd

    actual = test_days(*load_protocol()).set_index("date")["actual"]
    out = {}
    for rec in records:
        s = pd.Series(rec["pred"], index=rec["dates"])
        s = s[s.index.isin(days)]
        out[rec["id"]] = float(np.mean(np.abs(s.to_numpy() - actual.loc[s.index].to_numpy())))
    return out


def stage_c_tasks(stage_b: dict[str, dict], top: int = 3) -> list[dict]:
    records = list(stage_b.values())
    days = set.intersection(*(set(r["dates"]) for r in records))
    mae = common_days_mae(records, days)
    tasks = []
    for arch, fs in product(["lstm", "transformer"], ["v1", "v2"]):
        group = sorted((r for r in records if r["arch"] == arch and r["feature_set"] == fs), key=lambda r: mae[r["id"]])[:top]
        for rank, rec in enumerate(group, start=1):
            for seed in SEEDS_C:
                tasks.append(make(f"C_{arch}_{fs}_top{rank}_{hp_key(arch, rec['hp'])}_s{seed}", "C", arch, rec["hp"], fs, seed))
    return tasks


def main(workers: int = 10, pilot: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if pilot:
        tasks = [t for t in stage_a_tasks() if t["id"] in {"A_lstm_series1_v1_s42", "A_lstm_series1_v2_s42"}]
        run_parallel(tasks, neural_task, RESULTS, workers, init_worker_cpu, "series20 pilot")
        return
    run_parallel(stage_a_tasks(), neural_task, RESULTS, workers, init_worker_cpu, "series20 stage A")
    stage_b = run_parallel(stage_b_tasks(), neural_task, RESULTS, workers, init_worker_cpu, "series20 stage B")
    run_parallel(stage_c_tasks(stage_b), neural_task, RESULTS, workers, init_worker_cpu, "series20 stage C")
    print(f"\nDone: {len(read_done(RESULTS))} tasks in {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    main(args.workers, args.pilot)
