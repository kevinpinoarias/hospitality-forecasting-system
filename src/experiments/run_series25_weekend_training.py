"""
Series 25 - training the model with the weekends in mind.

Track A item 3 of the accuracy roadmap. Series 23-24 changed the forecasts AFTER the model made them
(a raw-£ Saturday model, then an adaptive correction). This series changes what the model LEARNS FROM:

  w1.5, w2, w3   one model on all days, with Friday/Saturday/Sunday rows counted 1.5, 2 or 3 times as heavily
  sepFS          a separate model trained only on Friday-to-Sunday rows, predicting those three days
  sepSat         a separate model trained only on Saturday rows, predicting Saturdays

Every variant is the final model's CatBoost recipe unchanged (38 features, depth 4, 300 iterations, learning
rate 0.05, 25 seeds, the 10 rolling-origin folds and 585 test days) with only the rows or their weights changed, and
each is run on BOTH targets (log1p and raw £), 10 runs. The separate models are NOT retuned for their smaller
training sets (about 156 weekend rows in the first fold, 400 by the last; about 52 and 120 Saturdays), so if
they do badly that may be the settings, not the idea; a smaller model would be a separate pre-registered step.

PRE-REGISTERED PLAN (written before any result was seen)
  baseline   the current setup on the weekend: Friday and Sunday from the log model, Saturday from the raw model
             (the Series 23 hybrid), seeds 0-24
  assembly   each variant is scored on the same 251 Friday-to-Sunday test days. Days a variant does not cover
             (Saturday only, or Friday-Sunday only models) are filled from the hybrid
  step 1     the winner is the variant with the lowest Friday-to-Sunday MAE on seeds 0-24
  step 2     refit the winner on 25 fresh seeds (100-124) and compare with the hybrid rebuilt from the fresh
             seeds already fitted in Series 24; the winner's settings are not re-chosen
  bar        kept only if, on the fresh seeds, (a) the 95% moving-block bootstrap interval of the Friday-to-Sunday
             MAE change excludes zero and is negative, (b) none of Friday, Saturday, Sunday is significantly worse
             (change interval entirely above zero), and (c) for a variant that predicts every day, Monday to
             Thursday is not significantly worse than the log model
  context    (not a selection) whether a passing winner still gains once the Series 24 correction (Saturday and
             Sunday, 8 weeks, strength 0.5) is applied on top of it

Run:  python -m src.experiments.run_series25_weekend_training [--workers 10]
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
from src.experiments.series19_21_common import _STATE, FINAL_SEEDS_25, OUT_DIR, init_worker, load_protocol, read_done, run_parallel, test_days
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL

RESULTS = OUT_DIR / "series25_weekend_training.jsonl"
RESULTS_FRESH = OUT_DIR / "series25_fresh_seeds.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series25_weekend_training{SUFFIX}"
FRESH_SEEDS = list(range(100, 125))
FRI_SUN = ("Friday", "Saturday", "Sunday")
WEEKEND_DOW = (4, 5, 6)  # pandas dayofweek: Friday, Saturday, Sunday
VARIANTS = {
    "w1.5": {"kind": "weight", "weight": 1.5, "label": "weekends x1.5"},
    "w2": {"kind": "weight", "weight": 2.0, "label": "weekends x2"},
    "w3": {"kind": "weight", "weight": 3.0, "label": "weekends x3"},
    "sepFS": {"kind": "subset", "dow": WEEKEND_DOW, "label": "separate Fri-Sun model"},
    "sepSat": {"kind": "subset", "dow": (5,), "label": "separate Saturday model"},
}
TARGETS = ("log1p", "raw")


# --------------------------------------------------------------------------- fitting


def fit_variant(df: pd.DataFrame, folds: list, variant: str, target: str, seeds: list[int]) -> np.ndarray:
    """Out-of-sample predictions in £ on every test day, shape (len(seeds), n_test_days). A separate model
    predicts only the days it covers; the other days are NaN."""
    from catboost import CatBoostRegressor

    spec = VARIANTS[variant]
    features = list(FINAL_FEATURES)
    out = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        dow_train, dow_test = train[DATE_COL].dt.dayofweek.to_numpy(), test[DATE_COL].dt.dayofweek.to_numpy()
        if spec["kind"] == "subset":
            fit_rows, cover = np.isin(dow_train, spec["dow"]), np.isin(dow_test, spec["dow"])
            weights = None
        else:
            fit_rows, cover = np.ones(len(train), bool), np.ones(len(test), bool)
            weights = np.where(np.isin(dow_train, WEEKEND_DOW), spec["weight"], 1.0)
        y = train[TARGET_COL].to_numpy()
        y = np.log1p(y) if target == "log1p" else y
        per_seed = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train.loc[fit_rows, features], y[fit_rows], sample_weight=None if weights is None else weights[fit_rows])
            pred = np.full(len(test), np.nan)
            raw = model.predict(test.loc[cover, features])
            pred[cover] = np.expm1(raw) if target == "log1p" else raw
            per_seed.append(pred)
        out.append(np.array(per_seed))
    return np.concatenate(out, axis=1)


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    per_seed = fit_variant(df, folds, task["variant"], task["target"], task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "target": task["target"], "seeds": task["seeds"],
            "per_seed": [[None if np.isnan(x) else round(float(x), 2) for x in row] for row in per_seed],
            "seconds": round(time.time() - started, 1)}


def make_tasks(variants: list[tuple[str, str]], seeds: list[int]) -> list[dict]:
    out = []
    for variant, target in variants:
        for i in range(0, len(seeds), s23.SEED_CHUNK):
            chunk = seeds[i:i + s23.SEED_CHUNK]
            out.append({"id": f"{variant}_{target}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "target": target, "seeds": chunk})
    return out


def all_variant_targets() -> list[tuple[str, str]]:
    return [(v, t) for v in VARIANTS for t in TARGETS]


def load_variant_bags(path, n_seeds: int, variant_targets: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    done = read_done(path)
    bags = {}
    for variant, target in variant_targets:
        rows = [np.array([[np.nan if x is None else x for x in r] for r in rec["per_seed"]])
                for rec in done.values() if rec["variant"] == variant and rec["target"] == target]
        per_seed = np.concatenate(rows, axis=0)
        assert per_seed.shape[0] == n_seeds, (variant, target, per_seed.shape)
        bags[(variant, target)] = per_seed.mean(axis=0)  # NaN stays NaN where a separate model does not predict
    return bags


# --------------------------------------------------------------------------- analysis


def assemble(variant_pred: np.ndarray, hybrid_pred: np.ndarray) -> np.ndarray:
    """The variant's forecasts where it has them, the hybrid's elsewhere."""
    return np.where(np.isnan(variant_pred), hybrid_pred, variant_pred)


def fri_sun_mask(weekday: np.ndarray) -> np.ndarray:
    return np.isin(weekday, FRI_SUN)


def variant_table(days: pd.DataFrame, hybrid_pred: np.ndarray, log_pred: np.ndarray, bags: dict) -> pd.DataFrame:
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    fs = fri_sun_mask(wd)
    mon_thu = ~fs
    rows = [{"variant": "current hybrid (baseline)", "target": "-", **{d: s24.mae(a, hybrid_pred, wd == d) for d in FRI_SUN},
             "Fri_to_Sun": s24.mae(a, hybrid_pred, fs), "Mon_to_Thu": s24.mae(a, hybrid_pred, mon_thu)}]
    for (variant, target), pred in bags.items():
        covers_all = not np.isnan(pred).any()
        full = assemble(pred, hybrid_pred)
        row = {"variant": VARIANTS[variant]["label"], "target": target, **{d: s24.mae(a, full, wd == d) for d in FRI_SUN},
               "Fri_to_Sun": s24.mae(a, full, fs),
               "Mon_to_Thu": s24.mae(a, pred, mon_thu) if covers_all else float("nan")}
        row["Fri_to_Sun_change_vs_hybrid"] = row["Fri_to_Sun"] - rows[0]["Fri_to_Sun"]
        row["Mon_to_Thu_change_vs_log"] = row["Mon_to_Thu"] - s24.mae(a, log_pred, mon_thu) if covers_all and target == "log1p" else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def winner_report(days: pd.DataFrame, hybrid_pred: np.ndarray, log_pred: np.ndarray, pred: np.ndarray, covers_all: bool) -> pd.DataFrame:
    """Change in MAE (variant minus hybrid) on Friday to Sunday and on each of those days, plus Monday to Thursday
    against the log model for a variant that predicts every day, with 95% block-bootstrap intervals."""
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    full = assemble(pred, hybrid_pred)
    diff = np.abs(full - a) - np.abs(hybrid_pred - a)
    rows = []
    segments = [("Friday to Sunday", fri_sun_mask(wd))] + [(d, wd == d) for d in FRI_SUN]
    for name, mask in segments:
        mean, lo, hi = s23.masked_block_bootstrap(diff, mask.astype(float))
        rows.append({"segment": name, "days": int(mask.sum()), "hybrid_MAE": s24.mae(a, hybrid_pred, mask), "variant_MAE": s24.mae(a, full, mask),
                     "change": mean, "lo": lo, "hi": hi})
    if covers_all:
        mask = ~fri_sun_mask(wd)
        mean, lo, hi = s23.masked_block_bootstrap(np.abs(pred - a) - np.abs(log_pred - a), mask.astype(float))
        rows.append({"segment": "Monday to Thursday (vs log model)", "days": int(mask.sum()), "hybrid_MAE": s24.mae(a, log_pred, mask),
                     "variant_MAE": s24.mae(a, pred, mask), "change": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def passes_bar(rep: pd.DataFrame) -> tuple[bool, bool, bool]:
    r = rep.set_index("segment")
    a = bool(r.loc["Friday to Sunday", "hi"] < 0)
    b = bool(all(r.loc[d, "lo"] <= 0 for d in FRI_SUN))
    c = True if "Monday to Thursday (vs log model)" not in r.index else bool(r.loc["Monday to Thursday (vs log model)", "lo"] <= 0)
    return a, b, c


def main(workers: int) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)

    # ---- step 1: the 10 runs on seeds 0-24
    run_parallel(make_tasks(all_variant_targets(), FINAL_SEEDS_25), fit_task, RESULTS, workers, init_worker, "series25")
    bags = load_variant_bags(RESULTS, len(FINAL_SEEDS_25), all_variant_targets())
    b0 = s23.load_bags()
    hybrid0 = s24.hybrid(wd, b0["log1p_final"], b0["raw_same_params"])
    table = variant_table(days, hybrid0, b0["log1p_final"], bags)
    table.to_csv(REPORT_DIR / "variants_seeds0_24.csv", index=False)
    print("\nStep 1 - Friday-to-Sunday MAE on seeds 0-24 (separate models filled in from the hybrid where they have no forecast):\n",
          table.round(1).to_string(index=False))
    ranked = table.iloc[1:].sort_values("Fri_to_Sun")
    best_label, best_target = ranked.iloc[0]["variant"], ranked.iloc[0]["target"]
    best_key = next(k for k in bags if VARIANTS[k[0]]["label"] == best_label and k[1] == best_target)
    print(f"\nWinner: {best_label}, {best_target} target: Friday-to-Sunday MAE {ranked.iloc[0]['Fri_to_Sun']:.1f} "
          f"against {table.iloc[0]['Fri_to_Sun']:.1f} for the hybrid (optimistic: it picked itself)")

    # ---- step 2: fresh seeds
    run_parallel(make_tasks([best_key], FRESH_SEEDS), fit_task, RESULTS_FRESH, workers, init_worker, "series25 fresh seeds")
    fresh_win = load_variant_bags(RESULTS_FRESH, len(FRESH_SEEDS), [best_key])[best_key]
    f = s24.load_bags(s24.RESULTS_FRESH, len(FRESH_SEEDS))
    hybrid_f = s24.hybrid(wd, f["log1p_final"], f["raw_same_params"])
    covers_all = not np.isnan(fresh_win).any()
    rep = winner_report(days, hybrid_f, f["log1p_final"], fresh_win, covers_all)
    rep.to_csv(REPORT_DIR / "winner_fresh_seeds.csv", index=False)
    print("\nStep 2 - the winner on FRESH seeds 100-124 (change = variant minus hybrid, 95% interval):\n", rep.round(1).to_string(index=False))
    ok_a, ok_b, ok_c = passes_bar(rep)
    print(f"\nBar: (a) Fri-Sun interval below zero: {ok_a}  (b) no weekend day significantly worse: {ok_b}  (c) Mon-Thu not worse: {ok_c}  ->  "
          f"{'PASSES' if ok_a and ok_b and ok_c else 'DOES NOT PASS'}")

    # ---- context: does it still help once the Series 24 correction is applied?
    sc = days["fold_id"].to_numpy() >= 1
    stacked_base = assemble(fresh_win, hybrid_f)
    stacked_base = np.where(fri_sun_mask(wd), stacked_base, hybrid_f)  # Monday to Thursday stay with the log model
    for name, base in (("hybrid + Series 24 correction", hybrid_f), ("winner + Series 24 correction", stacked_base)):
        c = s24.correct(days, base, 8, 0.5, scope=("Saturday", "Sunday"))
        print(f"Context: {name:32s} MAE on scored days {s24.mae(a, c, sc):.2f}   (uncorrected {s24.mae(a, base, sc):.2f})")
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args().workers)
