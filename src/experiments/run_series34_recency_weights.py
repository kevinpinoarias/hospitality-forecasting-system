"""
Series 34 - recency weighting: do recent days deserve more say than old ones?

Track C, step one. A boosted-tree model counts a day from 18 months ago exactly like yesterday and adapts to change only through its
features. The Saturday and first-week biases both drift, so a model that listens more to recent days might follow them. A hard cut-off
was tested in Series 7 (a sliding 365-day window, £924 against £908) and was worse, because dropping old days loses information;
here old days fade instead of disappearing.

THE WEIGHTS. Inside each rolling-origin fold, a training day that is `age` days older than the last training day gets the weight
0.5 ** (age / half_life), passed to CatBoost as `sample_weight`; nothing from the test block is involved. Weighting also shrinks the
EFFECTIVE sample size, (sum of weights) squared over the sum of squared weights, which matters with about 950 rows, so it is reported next
to every result.

  hl45, hl90, hl180, hl365, hl730   half-lives in days (45 added at the request of the project owner)
  <winner>_floor                    the winning half-life with no weight allowed below 0.2 (tried once the winner is known)
  control                           the unweighted models (the Series 23 fits on the verified calendar, `series19_21_cal2`)

Both targets (log1p and raw £), the final recipe (38 features, depth 4, 300 iterations, learning rate 0.05, 25 seeds, 10 folds,
585 test days), verified calendar.

PRE-REGISTERED PLAN (written before any result was seen)
  step 1     the five half-lives; the winner is the one with the largest improvement (most negative change) in the final (log) model's
             MAE over all 585 days against the control, on the same days
  step 1b    the winner's floor variant; whichever of the two improves more goes forward
  step 2     refit that variant on 25 fresh seeds (100-124) and compare with the control on the same seeds, not re-chosen
  bar        kept only if on the fresh seeds (a) the 95% moving-block bootstrap interval of the overall MAE change is entirely below zero,
             and (b) no weekday, no weekday group (Friday to Sunday) and no school phase is significantly worse (interval entirely above zero)
  reading    changes in MAE with intervals, never levels; the raw-£ model and the hybrid (raw Saturdays) are context
  next       Track C step two (weights together with the Saturday and Sunday correction) only if this passes or comes close

Run:  python -m src.experiments.run_series34_recency_weights [--workers 10]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_package_validation as pv
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series29_school_calendar as s29
from src.experiments.run_error_autopsy import school_phase
from src.experiments.series19_21_common import _STATE, OUT_DIR, SUFFIX, FINAL_SEEDS_25, load_protocol, read_done, run_parallel, test_days
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL

RESULTS = OUT_DIR / "series34_recency_weights.jsonl"
RESULTS_FRESH = OUT_DIR / "series34_fresh_seeds.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series34_recency_weights{SUFFIX}"
FRESH_SEEDS = list(range(100, 125))
HALF_LIVES = (45, 90, 180, 365, 730)
FLOOR = 0.2
TARGETS = ("log1p", "raw")
CONTROL_SUFFIX = "_cal2"  # the unweighted fits on the verified calendar
PHASES = ["last week of term", "first week of a break", "middle of a break", "last week of a break", "other term time"]


# --------------------------------------------------------------------------- the weights


def recency_weights(dates: pd.Series, last_day: pd.Timestamp, half_life: float, floor: float = 0.0) -> np.ndarray:
    """0.5 ** (age / half_life) for each training date, where age is the days before the last training day; never below `floor`."""
    age = (pd.Timestamp(last_day) - pd.to_datetime(dates)).dt.days.to_numpy().astype(float)
    return np.maximum(0.5 ** (age / half_life), floor)


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def variant_spec(name: str) -> tuple[int, float]:
    """'hl90' -> (90, 0.0); 'hl90_floor' -> (90, 0.2)."""
    base, _, tail = name.partition("_")
    return int(base[2:]), (FLOOR if tail == "floor" else 0.0)


# --------------------------------------------------------------------------- fitting


def fit_variant(df: pd.DataFrame, folds: list, variant: str, target: str, seeds: list[int]) -> tuple[np.ndarray, list[float]]:
    """Out-of-sample predictions in £ on every test day (one row per seed) and the effective sample size of each fold."""
    from catboost import CatBoostRegressor

    half_life, floor = variant_spec(variant)
    features = list(FINAL_FEATURES)
    out, ess = [], []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        w = recency_weights(train[DATE_COL], train[DATE_COL].max(), half_life, floor)
        ess.append(effective_sample_size(w))
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        per_seed = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y, sample_weight=w)
            pred = model.predict(test[features])
            per_seed.append(np.expm1(pred) if target == "log1p" else pred)
        out.append(np.array(per_seed))
    return np.concatenate(out, axis=1), ess


def init_worker34() -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = s29.with_calendar(df), folds


def fit_task(task: dict) -> dict:
    started = time.time()
    per_seed, ess = fit_variant(_STATE["df"], _STATE["folds"], task["variant"], task["target"], task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "target": task["target"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "ess": [round(e, 1) for e in ess],
            "seconds": round(time.time() - started, 1)}


def make_tasks(variants: list[str], seeds: list[int]) -> list[dict]:
    out = []
    for variant in variants:
        for target in TARGETS:
            for i in range(0, len(seeds), s23.SEED_CHUNK):
                chunk = seeds[i:i + s23.SEED_CHUNK]
                out.append({"id": f"{variant}_{target}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "target": target, "seeds": chunk})
    return out


def load_bags(path, n_seeds: int, variants: list[str]) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, float]]:
    done = read_done(path)
    bags, ess = {}, {}
    for variant in variants:
        for target in TARGETS:
            recs = [r for r in done.values() if r["variant"] == variant and r["target"] == target]
            per_seed = np.concatenate([np.array(r["per_seed"]) for r in recs], axis=0)
            assert per_seed.shape[0] == n_seeds, (variant, target, per_seed.shape)
            bags[(variant, target)] = per_seed.mean(axis=0)
            ess[variant] = float(np.mean(recs[0]["ess"]))
    return bags, ess


# --------------------------------------------------------------------------- analysis


def masks_for(days: pd.DataFrame) -> dict[str, np.ndarray]:
    wd = days["weekday"].to_numpy()
    phase = np.array([school_phase(pd.Timestamp(d), s29.LONG_BREAKS) for d in days["date"]])
    out = {"All 585 days": np.ones(len(wd), bool)}
    out.update({d: wd == d for d in s23.WEEKDAYS})
    out["Friday to Sunday"] = np.isin(wd, ["Friday", "Saturday", "Sunday"])
    out.update({p: phase == p for p in PHASES})
    return out


def overall_change(a: np.ndarray, control: np.ndarray, new: np.ndarray) -> tuple[float, float, float]:
    diff = np.abs(new - a) - np.abs(control - a)
    return s23.masked_block_bootstrap(diff, np.ones(len(a)))


def segment_table(a: np.ndarray, control: np.ndarray, new: np.ndarray, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame([s29.change_row(name, a, control, new, m) for name, m in masks.items()])


def passes_bar(table: pd.DataFrame) -> tuple[bool, bool]:
    r = table.set_index("segment")
    a = bool(r.loc["All 585 days", "hi"] < 0)
    b = bool(all(r.loc[s, "lo"] <= 0 for s in r.index if s != "All 585 days"))
    return a, b


def main(workers: int) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    masks = masks_for(days)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    control = pv.load_set(CONTROL_SUFFIX)
    ctl0, ctlf = control["log_raw_0_24"], control["log_raw_fresh"]

    # ---- step 1: the five half-lives
    names = [f"hl{h}" for h in HALF_LIVES]
    run_parallel(make_tasks(names, FINAL_SEEDS_25), fit_task, RESULTS, workers, init_worker34, "series34")
    bags, ess = load_bags(RESULTS, 25, names)
    hyb_ctl = s24.hybrid(wd, ctl0["log1p_final"], ctl0["raw_same_params"])
    rows = []
    for n in names:
        m, lo, hi = overall_change(a, ctl0["log1p_final"], bags[(n, "log1p")])
        rm, _, _ = overall_change(a, ctl0["raw_same_params"], bags[(n, "raw")])
        hm, hlo, hhi = overall_change(a, hyb_ctl, s24.hybrid(wd, bags[(n, "log1p")], bags[(n, "raw")]))
        rows.append({"variant": n, "effective_sample_size": ess[n], "log_change": m, "log_lo": lo, "log_hi": hi, "raw_change": rm,
                     "hybrid_change": hm, "hybrid_lo": hlo, "hybrid_hi": hhi})
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "half_lives_seeds0_24.csv", index=False)
    print("\nStep 1 - the weights against the unweighted control on the same 585 days, seeds 0-24 (change in MAE, £; negative is better;\n"
          "effective sample size = mean over folds, the unweighted training sets average about 635 rows):\n", table.round(2).to_string(index=False))
    winner = table.sort_values("log_change").iloc[0]["variant"]
    print(f"\nWinner (largest improvement of the final model): {winner}")

    # ---- step 1b: the winner with a floor
    floor_name = f"{winner}_floor"
    run_parallel(make_tasks([floor_name], FINAL_SEEDS_25), fit_task, RESULTS, workers, init_worker34, "series34 floor")
    fb, fess = load_bags(RESULTS, 25, [floor_name])
    fm, flo, fhi = overall_change(a, ctl0["log1p_final"], fb[(floor_name, "log1p")])
    print(f"{floor_name}: effective sample size {fess[floor_name]:.0f}, final model change {fm:+.2f} [{flo:+.2f}, {fhi:+.2f}]")
    best_name = floor_name if fm < float(table.set_index("variant").loc[winner, "log_change"]) else winner
    print(f"Goes forward: {best_name}")

    # ---- step 2: fresh seeds
    run_parallel(make_tasks([best_name], FRESH_SEEDS), fit_task, RESULTS_FRESH, workers, init_worker34, "series34 fresh seeds")
    nb, _ = load_bags(RESULTS_FRESH, 25, [best_name])
    seg = segment_table(a, ctlf["log1p_final"], nb[(best_name, "log1p")], masks)
    seg.to_csv(REPORT_DIR / "winner_fresh_seeds_log.csv", index=False)
    print(f"\nStep 2 - {best_name} on FRESH seeds 100-124, the final (log) model, change against the unweighted control (95% interval):\n", seg.round(1).to_string(index=False))
    ok_a, ok_b = passes_bar(seg)
    print(f"\nBar: (a) overall interval below zero: {ok_a}  (b) nothing significantly worse: {ok_b}  ->  {'PASSES' if ok_a and ok_b else 'DOES NOT PASS'}")
    raw_seg = segment_table(a, ctlf["raw_same_params"], nb[(best_name, "raw")], masks)
    hyb_c = s24.hybrid(wd, ctlf["log1p_final"], ctlf["raw_same_params"])
    hyb_n = s24.hybrid(wd, nb[(best_name, "log1p")], nb[(best_name, "raw")])
    hyb_seg = segment_table(a, hyb_c, hyb_n, masks)
    print("\nContext - the raw-£ model:\n", raw_seg.round(1).to_string(index=False))
    print("\nContext - the hybrid (raw Saturdays):\n", hyb_seg.round(1).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args().workers)
