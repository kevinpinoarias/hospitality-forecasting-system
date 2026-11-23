"""
Series 26 - features chosen for the weekends.

Track A item 4 of the accuracy roadmap. The final model sees only three recent-sales inputs (sales 7 days
ago, the 7- and 14-day averages). This series gives it inputs that matter most on a Friday to Sunday:

  family A, same-weekday memory (only past days, matched by calendar date):
      lag_14_sales, lag_21_sales          sales on the same weekday 2 and 3 weeks ago
      same_wd_mean4, same_wd_max4         the average and the highest of the last 4 same weekdays
  family B, weekend interactions (Friday to Sunday x a condition):
      wk_payday   x within 3 days of payday      wk_dry   x dry day
      wk_temp     x max temperature              wk_bh_near x within 3 days of a bank holiday
      (school-holiday interactions are left for the Track B calendar fix: the school calendar has gaps)

  A_all      the four family A features
  A_summary  same_wd_mean4 and same_wd_max4 only
  A_lags     lag_14_sales and lag_21_sales only
  B          the four family B interactions
  AB         all eight

Each variant appends its features to the final model's 38 (the original order is unchanged), is fit on BOTH
targets (log1p and raw £) with the final recipe (depth 4, 300 iterations, learning rate 0.05, 25 seeds, 10
folds, 585 test days; the new columns can be missing at the start of the data and CatBoost handles that),
and is then assembled into its own hybrid exactly like the current setup: the raw model on Saturdays, the log
model on every other day. 10 runs.

PRE-REGISTERED PLAN (written before any result was seen)
  baseline   the Series 23 hybrid, seeds 0-24
  step 1     the winner is the variant whose hybrid has the lowest Friday-to-Sunday MAE on seeds 0-24
  step 2     refit the winner on 25 fresh seeds (100-124) and compare with the hybrid rebuilt from the fresh
             seeds fitted in Series 24; settings are not re-chosen
  bar        kept only if on the fresh seeds (a) the 95% moving-block bootstrap interval of the
             Friday-to-Sunday MAE change is entirely below zero, (b) none of Friday, Saturday, Sunday is
             significantly worse and (c) Monday to Thursday is not significantly worse
  caveat     family A is made of recent sales, which the live API does not have for dates weeks ahead (it fills
             them with seasonal averages, Series 18), so a family A gain is a near-term gain until the
             forecasting-at-a-distance track is done; family B uses only calendar and weather
  context    (not a selection) whether a passing winner still gains with the Series 24 correction on top

Run:  python -m src.experiments.run_series26_weekend_features [--workers 10]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.series19_21_common import SUFFIX
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series25_weekend_training as s25
from src.experiments.series19_21_common import (
    _STATE, FINAL_SEEDS_25, OUT_DIR, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days,
)
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL

RESULTS = OUT_DIR / "series26_weekend_features.jsonl"
RESULTS_FRESH = OUT_DIR / "series26_fresh_seeds.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series26_weekend_features{SUFFIX}"
FRESH_SEEDS = list(range(100, 125))
FAMILY_A = ["lag_14_sales", "lag_21_sales", "same_wd_mean4", "same_wd_max4"]
FAMILY_B = ["wk_payday", "wk_dry", "wk_temp", "wk_bh_near"]
VARIANTS = {
    "A_all": {"extra": FAMILY_A, "label": "A: same-weekday memory (4)"},
    "A_summary": {"extra": ["same_wd_mean4", "same_wd_max4"], "label": "A: mean and max of last 4 (2)"},
    "A_lags": {"extra": ["lag_14_sales", "lag_21_sales"], "label": "A: sales 2 and 3 weeks ago (2)"},
    "B": {"extra": FAMILY_B, "label": "B: weekend interactions (4)"},
    "AB": {"extra": FAMILY_A + FAMILY_B, "label": "A + B (8)"},
}
TARGETS = ("log1p", "raw")


# --------------------------------------------------------------------------- the features


def add_weekend_features(frame: pd.DataFrame, daily_sales: pd.Series) -> pd.DataFrame:
    """Append the family A and B columns to `frame` (one row per day). `daily_sales` is the sales series
    indexed by calendar date over ALL days of the data (including the warm-up rows the model frame drops), so
    a lag is the sales on that calendar date, never a neighbouring row, and uses only days before the row."""
    s = daily_sales.asfreq("D")
    lags = pd.DataFrame({k: s.shift(k) for k in (7, 14, 21, 28)})
    extra = pd.DataFrame({"lag_14_sales": lags[14], "lag_21_sales": lags[21],
                          "same_wd_mean4": lags.mean(axis=1), "same_wd_max4": lags.max(axis=1)})
    out = frame.copy()
    for col in FAMILY_A:
        out[col] = extra[col].reindex(pd.DatetimeIndex(out[DATE_COL])).to_numpy()
    fri_sun = (out[DATE_COL].dt.dayofweek >= 4).astype(float)
    near_bank_holiday = (np.minimum(out["days_to_bank_holiday"], out["days_since_bank_holiday"]) <= 3).astype(float)
    out["wk_payday"] = fri_sun * out["is_payday_window_pm3"]
    out["wk_dry"] = fri_sun * out["is_dry_day"]
    out["wk_temp"] = fri_sun * out["max_temp"]
    out["wk_bh_near"] = fri_sun * near_bank_holiday
    return out


def load_frame() -> tuple[pd.DataFrame, list]:
    df, folds = load_protocol()
    full = pd.read_csv(CONFIG.path("features_dir") / "engineered_features.csv", parse_dates=[DATE_COL])
    sales = full.set_index(DATE_COL)[TARGET_COL].sort_index()
    return add_weekend_features(df, sales), folds


def init_worker26() -> None:
    _STATE["df"], _STATE["folds"] = load_frame()


# --------------------------------------------------------------------------- fitting


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    features = list(FINAL_FEATURES) + VARIANTS[task["variant"]]["extra"]
    if task["target"] == "log1p":
        per_seed = bagged_log1p_predictions(df, folds, features, FINAL_PARAMS, task["seeds"])
    else:
        per_seed = s23.raw_predictions(df, folds, features, FINAL_PARAMS, task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "target": task["target"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def all_variant_targets() -> list[tuple[str, str]]:
    return [(v, t) for v in VARIANTS for t in TARGETS]


def load_bags(path, n_seeds: int, variant_targets: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    done = read_done(path)
    bags = {}
    for variant, target in variant_targets:
        per_seed = np.concatenate([np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant and r["target"] == target], axis=0)
        assert per_seed.shape[0] == n_seeds, (variant, target, per_seed.shape)
        bags[(variant, target)] = per_seed.mean(axis=0)
    return bags


# --------------------------------------------------------------------------- analysis


def variant_hybrids(days: pd.DataFrame, bags: dict) -> dict[str, np.ndarray]:
    wd = days["weekday"].to_numpy()
    return {v: s24.hybrid(wd, bags[(v, "log1p")], bags[(v, "raw")]) for v in VARIANTS}


def variant_table(days: pd.DataFrame, baseline: np.ndarray, hybrids: dict[str, np.ndarray], bags: dict, b0: dict) -> pd.DataFrame:
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    fs = s25.fri_sun_mask(wd)

    def row(name, pred, log_pred=None, raw_pred=None):
        r = {"variant": name, **{d: s24.mae(a, pred, wd == d) for d in s25.FRI_SUN}, "Fri_to_Sun": s24.mae(a, pred, fs),
             "Mon_to_Thu": s24.mae(a, pred, ~fs), "All_days": s24.mae(a, pred, np.ones(len(a), bool))}
        if log_pred is not None:
            r["log_model_MAE"], r["raw_model_MAE"] = s24.mae(a, log_pred, np.ones(len(a), bool)), s24.mae(a, raw_pred, np.ones(len(a), bool))
        return r

    rows = [row("current hybrid (baseline)", baseline, b0["log1p_final"], b0["raw_same_params"])]
    for v, pred in hybrids.items():
        rows.append(row(VARIANTS[v]["label"], pred, bags[(v, "log1p")], bags[(v, "raw")]))
    table = pd.DataFrame(rows)
    table["Fri_to_Sun_change"] = table["Fri_to_Sun"] - table.loc[0, "Fri_to_Sun"]
    table["All_days_change"] = table["All_days"] - table.loc[0, "All_days"]
    return table


def main(workers: int) -> None:
    df, folds = load_frame()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)

    def tasks(variant_targets, seeds):
        out = []
        for variant, target in variant_targets:
            for i in range(0, len(seeds), s23.SEED_CHUNK):
                chunk = seeds[i:i + s23.SEED_CHUNK]
                out.append({"id": f"{variant}_{target}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "target": target, "seeds": chunk})
        return out

    # ---- step 1
    run_parallel(tasks(all_variant_targets(), FINAL_SEEDS_25), fit_task, RESULTS, workers, init_worker26, "series26")
    bags = load_bags(RESULTS, len(FINAL_SEEDS_25), all_variant_targets())
    b0 = s23.load_bags()
    baseline = s24.hybrid(wd, b0["log1p_final"], b0["raw_same_params"])
    hybrids = variant_hybrids(days, bags)
    table = variant_table(days, baseline, hybrids, bags, b0)
    table.to_csv(REPORT_DIR / "variants_seeds0_24.csv", index=False)
    print("\nStep 1 - each variant's hybrid (raw Saturdays, log elsewhere) on seeds 0-24, MAE in £:\n", table.round(1).to_string(index=False))
    best = min(hybrids, key=lambda v: s24.mae(a, hybrids[v], s25.fri_sun_mask(wd)))
    print(f"\nWinner: {VARIANTS[best]['label']}: Friday-to-Sunday MAE {s24.mae(a, hybrids[best], s25.fri_sun_mask(wd)):.1f} "
          f"against {s24.mae(a, baseline, s25.fri_sun_mask(wd)):.1f} (optimistic: it picked itself)")

    # ---- step 2
    run_parallel(tasks([(best, t) for t in TARGETS], FRESH_SEEDS), fit_task, RESULTS_FRESH, workers, init_worker26, "series26 fresh seeds")
    fb = load_bags(RESULTS_FRESH, len(FRESH_SEEDS), [(best, t) for t in TARGETS])
    f = s24.load_bags(s24.RESULTS_FRESH, len(FRESH_SEEDS))
    hybrid_f = s24.hybrid(wd, f["log1p_final"], f["raw_same_params"])
    win_f = s24.hybrid(wd, fb[(best, "log1p")], fb[(best, "raw")])
    rep = s25.winner_report(days, hybrid_f, f["log1p_final"], win_f, covers_all=True)
    rep.to_csv(REPORT_DIR / "winner_fresh_seeds.csv", index=False)
    print("\nStep 2 - the winner on FRESH seeds 100-124 (change = winner's hybrid minus the current hybrid, 95% interval):\n",
          rep.round(1).to_string(index=False))
    ok_a, ok_b, ok_c = s25.passes_bar(rep)
    print(f"\nBar: (a) Fri-Sun interval below zero: {ok_a}  (b) no weekend day significantly worse: {ok_b}  (c) Mon-Thu not worse: {ok_c}  ->  "
          f"{'PASSES' if ok_a and ok_b and ok_c else 'DOES NOT PASS'}")
    if any(c in VARIANTS[best]["extra"] for c in FAMILY_A):
        print("Serving caveat: the winner uses family A (recent sales), which the API does not have weeks ahead.")

    sc = days["fold_id"].to_numpy() >= 1
    for name, base in (("current hybrid + Series 24 correction", hybrid_f), ("winner's hybrid + Series 24 correction", win_f)):
        c = s24.correct(days, base, 8, 0.5, scope=("Saturday", "Sunday"))
        print(f"Context: {name:42s} MAE on scored days {s24.mae(a, c, sc):.2f}   (uncorrected {s24.mae(a, base, sc):.2f})")
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args().workers)
