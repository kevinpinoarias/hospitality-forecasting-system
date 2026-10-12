"""
Series 30 - features that describe where a day sits in a school break.

Track B, step two. With the school calendar verified (Series 29, docs/SCHOOL_CALENDAR.md) the first week of a break is
still the hardest stretch of the year (about £1,433 against £910 overall). The model only knows "is it a school holiday";
this series tells it where in the break the day falls. Everything comes from the calendar, so it is known months ahead.

Counters (all five break types: Christmas, Easter, summer, October week, February mid-term; blank when not applicable)
  tf_days_since_start   0 on the first pupil-free day of a break, 1 the next day, and so on; blank outside a break
  tf_days_to_end        days left until pupils return (0 on the last day of the break); blank outside a break
  tf_days_to_start      days until the next break starts; blank inside a break
Flags (the three long holidays only; for a 5-day break the whole break is its first week)
  tf_first_week         the first 7 days of a break
  tf_last_week_term     the 7 days before a break starts
  tf_last_week_break    the last 7 days of a break

  TF_all       all six
  TF_flags     the three flags
  TF_counters  the three counters
  TF_capped    the three counters with every value above 14 set to 14 (so only the nearness to the edge of a break
               is visible, not the long quiet middle)

Each variant appends its columns to the final model's 38 (original order unchanged) and is fit on BOTH targets (log1p and
raw £) with the final recipe (depth 4, 300 iterations, learning rate 0.05, 25 seeds, 10 folds, 585 test days) on the
verified calendar. 8 runs. The baseline is the same model on the verified calendar (Series 29, stage 2).

PRE-REGISTERED PLAN (written before any result was seen)
  primary    the log (final) model
  phases     a day's place in a break is the autopsy's: last week of term, first week of a break, middle of a break, last
             week of a break, other term time (long holidays only). The three transition phases pooled are about 105 days.
  step 1     the winner is the variant with the lowest MAE on the transition-phase days on seeds 0-24
  step 2     refit the winner on 25 fresh seeds (100-124) and compare with the baseline on the same seeds; settings not
             re-chosen
  bar        kept only if on the fresh seeds (a) the 95% moving-block bootstrap interval of the transition-days MAE change
             is entirely below zero, (b) the overall MAE is not significantly worse, and (c) none of the three transition
             phases, the middle of a break or ordinary term time is significantly worse
  context    (not a selection) the same change for the raw-£ model and for the hybrid (raw Saturdays)

Run:  python -m src.experiments.run_series30_transition_features [--workers 10]
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
from src.experiments import run_series29_school_calendar as s29
from src.experiments.run_error_autopsy import school_phase
from src.experiments.series19_21_common import (
    _STATE, FINAL_SEEDS_25, OUT_DIR, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days,
)
from src.features.build_features import LONG_BREAK_TYPES, SCHOOL_BREAKS
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS

RESULTS = OUT_DIR / "series30_transition_features.jsonl"
RESULTS_FRESH = OUT_DIR / "series30_fresh_seeds.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series30_transition_features{SUFFIX}"
FRESH_SEEDS = list(range(100, 125))
COUNTERS = ["tf_days_since_start", "tf_days_to_end", "tf_days_to_start"]
FLAGS = ["tf_first_week", "tf_last_week_term", "tf_last_week_break"]
CAPPED = ["tf_days_since_start_c14", "tf_days_to_end_c14", "tf_days_to_start_c14"]
CAP = 14
VARIANTS = {
    "TF_all": {"extra": COUNTERS + FLAGS, "label": "all six"},
    "TF_flags": {"extra": FLAGS, "label": "three flags"},
    "TF_counters": {"extra": COUNTERS, "label": "three counters"},
    "TF_capped": {"extra": CAPPED, "label": "counters capped at 14"},
}
TARGETS = ("log1p", "raw")
TRANSITION_PHASES = ["last week of term", "first week of a break", "last week of a break"]
OTHER_PHASES = ["middle of a break", "other term time"]


# --------------------------------------------------------------------------- the features


def transition_features(dates: pd.Series, breaks: pd.DataFrame = SCHOOL_BREAKS) -> pd.DataFrame:
    """The six transition columns plus the three capped counters for each date. Depends only on the date and the calendar."""
    spans = breaks.copy()
    spans["start"], spans["end"] = pd.to_datetime(spans["start"]), pd.to_datetime(spans["end"])
    spans = spans.sort_values("start").reset_index(drop=True)
    rows = []
    for d in pd.to_datetime(dates).dt.normalize():
        inside = spans[(spans["start"] <= d) & (d <= spans["end"])]
        row = {c: np.nan for c in COUNTERS}
        row.update({c: 0 for c in FLAGS})
        if len(inside):
            b = inside.iloc[0]
            row["tf_days_since_start"], row["tf_days_to_end"] = (d - b["start"]).days, (b["end"] - d).days
            if b["break_type"] in LONG_BREAK_TYPES:
                row["tf_first_week"] = int(row["tf_days_since_start"] < 7)
                row["tf_last_week_break"] = int(row["tf_days_to_end"] < 7)
        else:
            later = spans[spans["start"] > d]
            if len(later):
                row["tf_days_to_start"] = (later.iloc[0]["start"] - d).days
            ahead = later[later["break_type"].isin(LONG_BREAK_TYPES)]
            if len(ahead):
                row["tf_last_week_term"] = int((ahead.iloc[0]["start"] - d).days <= 7)
        rows.append(row)
    out = pd.DataFrame(rows)
    for plain, capped in zip(COUNTERS, CAPPED):
        out[capped] = out[plain].clip(upper=CAP)
    return out


def with_transition(frame: pd.DataFrame) -> pd.DataFrame:
    new = frame.copy()
    feats = transition_features(new[DATE_COL])
    for col in feats.columns:
        new[col] = feats[col].to_numpy()
    return new


def init_worker30() -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = with_transition(s29.with_calendar(df)), folds


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


def make_tasks(variant_targets: list[tuple[str, str]], seeds: list[int]) -> list[dict]:
    out = []
    for variant, target in variant_targets:
        for i in range(0, len(seeds), s23.SEED_CHUNK):
            chunk = seeds[i:i + s23.SEED_CHUNK]
            out.append({"id": f"{variant}_{target}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "target": target, "seeds": chunk})
    return out


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


def segment_masks(phase: np.ndarray) -> dict[str, np.ndarray]:
    out = {"All 585 days": np.ones(len(phase), bool), "Transition weeks (3 phases)": np.isin(phase, TRANSITION_PHASES)}
    out.update({p: phase == p for p in TRANSITION_PHASES + OTHER_PHASES})
    return out


def variant_table(actual: np.ndarray, phase: np.ndarray, baseline: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    masks = segment_masks(phase)
    rows = []
    for name, pred in {"baseline (verified calendar)": baseline, **preds}.items():
        row = {"variant": name}
        for seg, m in masks.items():
            row[seg] = float(np.abs(pred - actual)[m].mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    t["transition_change"] = t["Transition weeks (3 phases)"] - t.loc[0, "Transition weeks (3 phases)"]
    t["overall_change"] = t["All 585 days"] - t.loc[0, "All 585 days"]
    return t


def change_table(actual: np.ndarray, phase: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([s29.change_row(name, actual, baseline, pred, m) for name, m in segment_masks(phase).items()])


def passes_bar(rep: pd.DataFrame) -> tuple[bool, bool, bool]:
    r = rep.set_index("segment")
    a = bool(r.loc["Transition weeks (3 phases)", "hi"] < 0)
    b = bool(r.loc["All 585 days", "lo"] <= 0)
    c = bool(all(r.loc[p, "lo"] <= 0 for p in TRANSITION_PHASES + OTHER_PHASES))
    return a, b, c


def main(workers: int) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    phase = np.array([school_phase(pd.Timestamp(d), s29.LONG_BREAKS) for d in days["date"]])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    base = s29.load_new_bags(s29.results_path("stage2"))

    run_parallel(make_tasks(all_variant_targets(), FINAL_SEEDS_25), fit_task, RESULTS, workers, init_worker30, "series30")
    bags = load_bags(RESULTS, 25, all_variant_targets())
    log_preds = {VARIANTS[v]["label"]: bags[(v, "log1p")] for v in VARIANTS}
    table = variant_table(a, phase, base["seeds0_24"]["log1p_final"], log_preds)
    table.to_csv(REPORT_DIR / "variants_seeds0_24.csv", index=False)
    print("\nStep 1 - the final (log) model with each variant, seeds 0-24, MAE £ (days: "
          f"{ {k: int(m.sum()) for k, m in segment_masks(phase).items()} }):\n", table.round(1).to_string(index=False))
    best_label = table.iloc[1:].sort_values("Transition weeks (3 phases)").iloc[0]["variant"]
    best = next(v for v in VARIANTS if VARIANTS[v]["label"] == best_label)
    print(f"\nWinner (lowest MAE on the transition weeks): {best_label}")

    run_parallel(make_tasks([(best, t) for t in TARGETS], FRESH_SEEDS), fit_task, RESULTS_FRESH, workers, init_worker30, "series30 fresh seeds")
    fb = load_bags(RESULTS_FRESH, 25, [(best, t) for t in TARGETS])
    rep = change_table(a, phase, base["seeds100_124"]["log1p_final"], fb[(best, "log1p")])
    rep.to_csv(REPORT_DIR / "winner_fresh_seeds.csv", index=False)
    print("\nStep 2 - the winner on FRESH seeds 100-124 against the baseline on the same seeds (log model; 95% interval):\n", rep.round(1).to_string(index=False))
    ok_a, ok_b, ok_c = passes_bar(rep)
    print(f"\nBar: (a) transition weeks interval below zero: {ok_a}  (b) overall not worse: {ok_b}  (c) no other phase worse: {ok_c}  ->  "
          f"{'PASSES' if ok_a and ok_b and ok_c else 'DOES NOT PASS'}")

    raw_rep = change_table(a, phase, base["seeds100_124"]["raw_same_params"], fb[(best, "raw")])
    hyb_base = s24.hybrid(wd, base["seeds100_124"]["log1p_final"], base["seeds100_124"]["raw_same_params"])
    hyb_new = s24.hybrid(wd, fb[(best, "log1p")], fb[(best, "raw")])
    hyb_rep = change_table(a, phase, hyb_base, hyb_new)
    print("\nContext - raw-£ model:\n", raw_rep.round(1).to_string(index=False))
    print("\nContext - hybrid (raw Saturdays):\n", hyb_rep.round(1).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args().workers)
