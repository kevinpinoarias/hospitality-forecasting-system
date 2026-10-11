"""
Shared machinery for Series 19-21 (column order, neural networks on the
38-feature set, CatBoost depth/iterations).

Every test here is scored under exactly the final model's own protocol - the
same 950 complete rows, the same 10 rolling-origin folds (365 days of initial
history, then back-to-back 60-day test blocks, expanding window) and the same
585 test days - so every figure is directly comparable with the final model's
£909.57 MAE (reports/results/final_model_metrics.csv).

Work is spread across processes (one task per worker call) and each finished
task is appended to a JSONL file, so a run can be stopped and resumed: a task
whose id is already in the file is skipped. CatBoost is deterministic whatever
its thread count (checked: predictions identical with 1 and many threads), so
running each fit single-threaded in parallel workers changes no result.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation.metrics import regression_metrics
from src.experiments.splits import generate_rolling_origin_folds
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL  # noqa: F401 (FINAL_PARAMS is re-exported to the series scripts)
from src.models.train_final_model import INITIAL_TRAIN_DAYS, TEST_DAYS, load_final_model_data

OUT_DIR = CONFIG.path("experiments_dir") / "series19_21"
FINAL_SEEDS_25 = list(range(25))  # the seeds the served model is bagged over
V1_FEATURES_ORDERED = FINAL_FEATURES[:15]

_STATE: dict = {}


# --------------------------------------------------------------------------- protocol


def load_protocol() -> tuple[pd.DataFrame, list]:
    """The final model's rows and folds."""
    df = load_final_model_data()
    folds = generate_rolling_origin_folds(
        df[DATE_COL], initial_train_days=INITIAL_TRAIN_DAYS, test_days=TEST_DAYS, step_days=TEST_DAYS
    )
    return df, folds


def test_days(df: pd.DataFrame, folds: list) -> pd.DataFrame:
    """date, fold_id, actual and the manager's forecast for every test day, in protocol order."""
    parts = []
    for fold in folds:
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        parts.append(pd.DataFrame({
            "date": test[DATE_COL].dt.strftime("%Y-%m-%d").to_numpy(), "fold_id": fold.fold_id,
            "actual": test[TARGET_COL].to_numpy(), "manual": test["forecast_sales"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def init_worker(hide_gpu: bool = False) -> None:
    if hide_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # the networks are tiny; CPU workers beat sharing one GPU
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = df, folds


def init_worker_cpu() -> None:
    init_worker(hide_gpu=True)


# --------------------------------------------------------------------------- scoring


def score(actual: np.ndarray, pred: np.ndarray, fold_ids: np.ndarray) -> dict:
    """Pooled and fold-mean metrics, as the final model reports them."""
    pooled = regression_metrics(pd.Series(actual), pd.Series(pred))
    fold_mae = [
        regression_metrics(pd.Series(actual[fold_ids == f]), pd.Series(pred[fold_ids == f]))["MAE"]
        for f in sorted(set(fold_ids.tolist()))
    ]
    return {**pooled, "MAE_fold_mean": float(np.mean(fold_mae)), "per_fold_MAE": [float(x) for x in fold_mae]}


# --------------------------------------------------------------------------- CatBoost task


def bagged_log1p_predictions(
    df: pd.DataFrame, folds: list, features: list[str], params: dict, seeds: list[int], thread_count: int = 1
) -> np.ndarray:
    """Out-of-sample predictions on every test day, one row per seed: the final
    model's exact recipe (log1p target, CatBoost, expm1 on the way back) with
    `features` in the given ORDER. Shape (len(seeds), n_test_days)."""
    from catboost import CatBoostRegressor

    out = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        target = np.log1p(train[TARGET_COL])
        per_seed = []
        for seed in seeds:
            model = CatBoostRegressor(
                random_state=seed, verbose=False, allow_writing_files=False, thread_count=thread_count, **params
            )
            model.fit(train[features], target)
            per_seed.append(np.expm1(model.predict(test[features])))
        out.append(np.array(per_seed))
    return np.concatenate(out, axis=1)


def catboost_task(task: dict) -> dict:
    """task: id, kind, features (ordered list), params, seeds."""
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    per_seed = bagged_log1p_predictions(df, folds, task["features"], task["params"], task["seeds"])
    days = test_days(df, folds)
    actual, fold_ids = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    bag = per_seed.mean(axis=0)
    record = {
        "id": task["id"], "kind": task["kind"], "label": task.get("label", task["id"]),
        "params": task["params"], "seeds": [task["seeds"][0], task["seeds"][-1], len(task["seeds"])],
        "features": task["features"],
        "bag": score(actual, bag, fold_ids),
        "seed1": score(actual, per_seed[0], fold_ids),
        "bag5": score(actual, per_seed[: min(5, len(per_seed))].mean(axis=0), fold_ids),
        "seconds": round(time.time() - started, 1),
        "pred_bag": [round(float(x), 2) for x in bag],
    }
    return record


# --------------------------------------------------------------------------- neural task


def neural_fold_predictions(
    df: pd.DataFrame, folds: list, features: list[str], arch: str, hp: dict, seed: int, include_target_day: bool = False
) -> pd.DataFrame:
    """The repo's own LSTM / Transformer, trained from scratch on each fold's
    expanding history and scored on that fold's test block. Like the originals,
    the network reads only the `window` days BEFORE the day it predicts (not the
    target day's own features) and skips windows that span a missing calendar
    day, so a few test days get no prediction; those days are reported."""
    from src.models.neural_common import predict, prepare_datasets, set_seed, train_model
    from src.models.train_lstm import LSTMForecaster
    from src.models.train_transformer import TransformerForecaster

    import torch

    torch.set_num_threads(1)
    frames = []
    for fold in folds:
        combined = df[df[DATE_COL] < fold.test_end].reset_index(drop=True)
        set_seed(seed)
        data = prepare_datasets(
            split_date=str(fold.test_start.date()), window=hp["window"], features=features, df=combined,
            train_window_days=None, valid_end_date=str(fold.test_end.date()), include_target_day=include_target_day,
        )
        if arch == "lstm":
            model = LSTMForecaster(n_features=data["n_features"], hidden_size=hp["hidden_size"], num_layers=hp["num_layers"])
        else:
            kwargs = {k: hp[k] for k in ("d_model", "nhead") if k in hp}
            model = TransformerForecaster(n_features=data["n_features"], window=hp["window"], **kwargs)
        train_model(model, data["train_dataset"], epochs=hp["epochs"], batch_size=16, lr=1e-3)
        pred = predict(model, data["valid_dataset"], data["scaler"])
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(data["valid_dates"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "fold_id": fold.fold_id, "actual": data["y_valid_actual"], "pred": pred,
        }))
    return pd.concat(frames, ignore_index=True)


def neural_task(task: dict) -> dict:
    """task: id, kind, arch, hp, features (ordered list), feature_set, seed."""
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    preds = neural_fold_predictions(
        df, folds, task["features"], task["arch"], task["hp"], task["seed"], task.get("include_target_day", False)
    )
    s = score(preds["actual"].to_numpy(), preds["pred"].to_numpy(), preds["fold_id"].to_numpy())
    return {
        "id": task["id"], "kind": task["kind"], "arch": task["arch"], "hp": task["hp"],
        "feature_set": task["feature_set"], "n_features": len(task["features"]), "seed": task["seed"],
        "include_target_day": task.get("include_target_day", False),
        "score": s, "n_days": len(preds), "seconds": round(time.time() - started, 1),
        "dates": preds["date"].tolist(), "pred": [round(float(x), 2) for x in preds["pred"]],
    }


# --------------------------------------------------------------------------- parallel runner


def read_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def run_parallel(tasks: list[dict], worker, path: Path, workers: int, initializer=init_worker, label: str = "") -> dict[str, dict]:
    """Run `tasks` (each a dict with a unique "id") on `workers` processes,
    appending each result to `path` as it finishes; tasks already in `path` are skipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    done = read_done(path)
    pending = [t for t in tasks if t["id"] not in done]
    print(f"[{label}] {len(tasks)} tasks, {len(done)} already done, {len(pending)} to run on {workers} workers", flush=True)
    if pending:
        started, finished = time.time(), 0
        with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
            futures = {pool.submit(worker, t): t["id"] for t in pending}
            for fut in as_completed(futures):
                rec = fut.result()
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done[rec["id"]] = rec
                finished += 1
                elapsed = time.time() - started
                eta = elapsed / finished * (len(pending) - finished)
                print(f"[{label}] {finished}/{len(pending)} done ({rec['id']}, {rec['seconds']}s) - "
                      f"elapsed {elapsed / 60:.1f} min, ~{eta / 60:.1f} min left", flush=True)
    return {i: done[i] for i in ids}
