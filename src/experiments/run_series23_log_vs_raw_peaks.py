"""
Series 23 - does the log transform hold back the busy days?

The final model predicts log1p(sales) and converts back with expm1. A model trained
on logs aims at the typical day rather than the average one, which can pull the very
busy days down. The autopsy found Saturdays about £720 too low and the biggest misses
on the busiest days, so this series only MEASURES whether the log target is part of
the reason: the final model against the same CatBoost trained on raw pounds.

  log1p_final         the final model's recipe (log1p target, depth 4, 300 iterations, lr 0.05)
  raw_same_params     the same settings, raw-£ target
  raw_series8_params  raw-£ target with the settings that won on the raw scale in Series 1/2/8
                      (depth 4, 300 iterations, lr 0.03), so the raw side is not handicapped

All three: 25 seeds (0-24), 38 features, the 10 rolling-origin folds and 585 test days of
the final model's protocol. Scored on Saturdays, Friday to Sunday, the top 10% of days by
actual sales and by the manager's forecast, and by decile of actual sales.

Two cautions are built into the analysis:
  * Picking the busiest days BY ACTUAL SALES makes any noisy model look under-forecast
    (the days were chosen for being higher than usual), so the fair reading is the
    DIFFERENCE between the two models on the same days, and the same table is repeated
    with days picked by the manager's forecast, which is known in advance.
  * The slope of actual on forecast needs no selection on the outcome: 1.0 means no
    shrinkage, above 1.0 means the forecasts are compressed towards the middle.

Run:  python -m src.experiments.run_series23_log_vs_raw_peaks [--workers 10] [--pilot]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.series19_21_common import SUFFIX
from src.experiments.series19_21_common import (
    _STATE, FINAL_PARAMS, FINAL_SEEDS_25, OUT_DIR, bagged_log1p_predictions, init_worker, read_done, run_parallel,
    score, test_days,
)
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

RESULTS = OUT_DIR / "series23_log_vs_raw.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series23_log_vs_raw_peaks{SUFFIX}"
SEED_CHUNK = 5
RAW_SERIES8_PARAMS = {**FINAL_PARAMS, "learning_rate": 0.03}
VARIANTS = {
    "log1p_final": {"target": "log1p", "params": dict(FINAL_PARAMS), "label": "Log1p target (final model)"},
    "raw_same_params": {"target": "raw", "params": dict(FINAL_PARAMS), "label": "Raw £ target, same settings"},
    "raw_series8_params": {"target": "raw", "params": RAW_SERIES8_PARAMS, "label": "Raw £ target, Series 8 settings"},
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --------------------------------------------------------------------------- fitting


def raw_predictions(df: pd.DataFrame, folds: list, features: list[str], params: dict, seeds: list[int]) -> np.ndarray:
    """Like bagged_log1p_predictions but trained on the raw sales: one row per seed."""
    from catboost import CatBoostRegressor

    out = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        per_seed = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **params)
            model.fit(train[features], train[TARGET_COL])
            per_seed.append(model.predict(test[features]))
        out.append(np.array(per_seed))
    return np.concatenate(out, axis=1)


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    spec = VARIANTS[task["variant"]]
    if spec["target"] == "log1p":
        per_seed = bagged_log1p_predictions(df, folds, list(FINAL_FEATURES), spec["params"], task["seeds"])
    else:
        per_seed = raw_predictions(df, folds, list(FINAL_FEATURES), spec["params"], task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def tasks() -> list[dict]:
    out = []
    for variant in VARIANTS:
        for i in range(0, len(FINAL_SEEDS_25), SEED_CHUNK):
            seeds = FINAL_SEEDS_25[i:i + SEED_CHUNK]
            out.append({"id": f"{variant}_s{seeds[0]}-{seeds[-1]}", "variant": variant, "seeds": seeds})
    return out


def load_bags() -> dict[str, np.ndarray]:
    """variant -> the 25-seed bag (mean of the per-seed predictions in pounds), shape (585,)."""
    done = read_done(RESULTS)
    bags = {}
    for variant in VARIANTS:
        rows = [np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant]
        per_seed = np.concatenate(rows, axis=0)
        assert per_seed.shape[0] == len(FINAL_SEEDS_25), (variant, per_seed.shape)
        bags[variant] = per_seed.mean(axis=0)
    return bags


# --------------------------------------------------------------------------- analysis helpers


def masked_block_bootstrap(diff: np.ndarray, mask: np.ndarray, block: int = 10, reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `diff` over the days in `mask`, with a 95% interval from a moving-block bootstrap
    of the WHOLE series (so the number of segment days varies by resample, as it would in a new sample)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    num = (diff * mask)[idx].sum(axis=1)
    den = mask[idx].sum(axis=1)
    ok = den > 0
    means = num[ok] / den[ok]
    return float((diff * mask).sum() / mask.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def slope(actual: np.ndarray, pred: np.ndarray) -> float:
    """Least-squares slope of actual on forecast. 1.0 = no shrinkage; above 1.0 = forecasts compressed."""
    if len(actual) < 3 or np.std(pred) == 0:
        return float("nan")
    return float(np.polyfit(pred, actual, 1)[0])


def segments(days: pd.DataFrame) -> dict[str, np.ndarray]:
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    wd = days["weekday"].to_numpy()
    top = lambda x, q: x >= np.quantile(x, 1 - q)  # noqa: E731
    sat = wd == "Saturday"
    sat_top = np.zeros(len(a), bool)
    sat_top[np.where(sat)[0][a[sat] >= np.quantile(a[sat], 0.8)]] = True
    return {
        "All test days": np.ones(len(a), bool),
        "Saturdays": sat,
        "Friday to Sunday": np.isin(wd, ["Friday", "Saturday", "Sunday"]),
        "Monday to Thursday": np.isin(wd, WEEKDAYS[:4]),
        "Top 10% of days by ACTUAL sales": top(a, 0.10),
        "Top 20% of days by ACTUAL sales": top(a, 0.20),
        "Top 20% of Saturdays by actual sales": sat_top,
        "Top 10% of days by MANAGER's forecast": top(m, 0.10),
        "Top 20% of days by MANAGER's forecast": top(m, 0.20),
    }


def segment_table(days: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    a = days["actual"].to_numpy()
    seg = segments(days)
    log_err = np.abs(preds["log1p_final"] - a)
    rows = []
    for name, mask in seg.items():
        row = {"segment": name, "days": int(mask.sum()), "mean_actual": float(a[mask].mean()),
               "manager_MAE": float(np.abs(days["manual"].to_numpy()[mask] - a[mask]).mean())}
        for v, p in preds.items():
            row[f"{v}_MAE"] = float(np.abs(p[mask] - a[mask]).mean())
            row[f"{v}_bias"] = float((p[mask] - a[mask]).mean())
            row[f"{v}_slope"] = slope(a[mask], p[mask]) if not name.startswith("Top") else float("nan")
        for v in ("raw_same_params", "raw_series8_params"):
            mean, lo, hi = masked_block_bootstrap(log_err - np.abs(preds[v] - a), mask.astype(float))
            row[f"log_minus_{v}_MAE"], row[f"log_minus_{v}_lo"], row[f"log_minus_{v}_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def decile_table(days: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Bias and MAE by decile of ACTUAL sales (decile 10 = the busiest 10% of days). Read with care:
    every noisy model is under-forecast in the top decile and over-forecast in the bottom one."""
    a = days["actual"].to_numpy()
    d = pd.qcut(pd.Series(a), 10, labels=False) + 1
    rows = []
    for k in range(1, 11):
        m = (d == k).to_numpy()
        row = {"decile_of_actual": k, "days": int(m.sum()), "mean_actual": float(a[m].mean())}
        for v, p in preds.items():
            row[f"{v}_bias"] = float((p[m] - a[m]).mean())
            row[f"{v}_MAE"] = float(np.abs(p[m] - a[m]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def weekday_table(days: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    a = days["actual"].to_numpy()
    rows = []
    for wd in WEEKDAYS:
        m = (days["weekday"] == wd).to_numpy()
        row = {"weekday": wd, "days": int(m.sum())}
        for v, p in preds.items():
            row[f"{v}_MAE"] = float(np.abs(p[m] - a[m]).mean())
            row[f"{v}_bias"] = float((p[m] - a[m]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(days: pd.DataFrame, preds: dict[str, np.ndarray], deciles: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"log1p_final": "#1a5fb4", "raw_same_params": "#c64600", "raw_series8_params": "#2f8f4e"}
    a = days["actual"].to_numpy()
    names = []

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for v in preds:
        ax.plot(deciles["decile_of_actual"], deciles[f"{v}_bias"], marker="o", label=VARIANTS[v]["label"], color=colours[v])
    ax.axhline(0, color="#888", lw=1)
    ax.set_xlabel("decile of actual daily sales (10 = the busiest 10% of days)")
    ax.set_ylabel("average forecast minus actual (£)")
    ax.set_title("Bias by how busy the day turned out to be")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "bias_by_decile.png", dpi=130); plt.close(fig)
    names.append("bias_by_decile.png")

    sat = (days["weekday"] == "Saturday").to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    lim = [a.min() * 0.9, a.max() * 1.05]
    for ax, v in zip(axes, ("log1p_final", "raw_same_params")):
        ax.scatter(preds[v][~sat], a[~sat], s=9, alpha=0.35, color="#999", label="other days")
        ax.scatter(preds[v][sat], a[sat], s=16, alpha=0.8, color=colours[v], label="Saturdays")
        ax.plot(lim, lim, color="#444", lw=1)
        ax.set_title(VARIANTS[v]["label"], fontsize=10)
        ax.set_xlabel("forecast (£)")
    axes[0].set_ylabel("actual sales (£)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Saturdays above the diagonal were under-forecast", fontsize=10)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "saturday_scatter.png", dpi=130); plt.close(fig)
    names.append("saturday_scatter.png")
    return names


def analyse() -> None:
    from src.experiments.series19_21_common import load_protocol

    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    bags = load_bags()
    a, f = days["actual"].to_numpy(), days["fold_id"].to_numpy()

    overall = pd.DataFrame([{"model": VARIANTS[v]["label"], **{k: score(a, p, f)[k] for k in ("MAE", "MAE_fold_mean", "RMSE", "Bias")}}
                            for v, p in bags.items()])
    print("\nOverall (pooled over 585 days):\n", overall.round(2).to_string(index=False))
    if SUFFIX == "":  # the original results set (original school calendar) must reproduce the logged figure exactly
        assert abs(overall.iloc[0]["MAE"] - 909.57) < 0.01, "the log model must reproduce the final model's £909.57"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    seg, dec, wdt = segment_table(days, bags), decile_table(days, bags), weekday_table(days, bags)
    overall.to_csv(REPORT_DIR / "overall.csv", index=False)
    seg.to_csv(REPORT_DIR / "segments.csv", index=False)
    dec.to_csv(REPORT_DIR / "deciles.csv", index=False)
    wdt.to_csv(REPORT_DIR / "weekdays.csv", index=False)
    make_figures(days, bags, dec)
    pd.DataFrame({**days, **{v: p for v, p in bags.items()}}).to_csv(REPORT_DIR / "predictions.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\nBy segment (MAE, bias = forecast minus actual):\n", seg[["segment", "days", "mean_actual", "manager_MAE", "log1p_final_MAE", "raw_same_params_MAE",
                                                                      "raw_series8_params_MAE", "log1p_final_bias", "raw_same_params_bias", "raw_series8_params_bias"]].round(1).to_string(index=False))
    print("\nLog minus raw (same params) MAE, 95% block-bootstrap interval:\n",
          seg[["segment", "log_minus_raw_same_params_MAE", "log_minus_raw_same_params_lo", "log_minus_raw_same_params_hi"]].round(1).to_string(index=False))
    print("\nSlope of actual on forecast (1.0 = no shrinkage):\n", seg[["segment", "log1p_final_slope", "raw_same_params_slope", "raw_series8_params_slope"]].round(3).to_string(index=False))
    print("\nBy decile of actual:\n", dec.round(0).to_string(index=False))
    print("\nBy weekday:\n", wdt.round(1).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


def main(workers: int, pilot: bool) -> None:
    todo = tasks()[:2] if pilot else tasks()
    run_parallel(todo, fit_task, RESULTS, workers, init_worker, "series23")
    if not pilot:
        analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    if args.analyse_only:
        analyse()
    else:
        main(args.workers, args.pilot)
