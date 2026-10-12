"""
Series 27 - each weekend day gets its own feature set.

Follow-up to Series 26 (Track A item 4). The same-weekday features helped the weekend and hurt Monday to Thursday,
so here Monday to Thursday keep the current hybrid and each of Friday, Saturday and Sunday is given the feature
option that suits it best, chosen separately. Options (the Series 26 variants, plus "none"):

  none, A_all, A_summary, A_lags, B, AB

each on the day's own target as in the current hybrid: the log model for Friday and Sunday, the raw model for
Saturday. Each weekend day has only about 84 test days, so choosing among 6 options on them is noisy and the choice
is tested on days it never saw.

PRE-REGISTERED PLAN (written before any result was seen)
  split      the 10 test blocks are cut in two: blocks 0-4 (selection) and blocks 5-9 (evaluation)
  step 1     for each of Friday, Saturday, Sunday, pick the option with the lowest MAE on that weekday's days in
             blocks 0-4 (seeds 0-24); ties go to "none"
  step 2     assemble the weekend (each day with its chosen option, Monday to Thursday = the current hybrid) and
             score it against the current hybrid on blocks 5-9, on the original seeds and on 25 fresh seeds
             (100-124, refitting only the options chosen)
  bar        kept only if on the fresh seeds and blocks 5-9 (a) the 95% moving-block bootstrap interval of the
             Friday-to-Sunday MAE change is entirely below zero and (b) none of Friday, Saturday, Sunday is
             significantly worse (Monday to Thursday are unchanged by construction)
  context    (not a selection) the same assembly with the choices made on all 585 days, which is optimistic; and
             the Series 24 correction applied on top of the chosen assembly
  caveat     options with the same-weekday features (A_*) use recent sales, which the live API does not have for
             dates weeks ahead, so a gain from them is a near-term gain

Run:  python -m src.experiments.run_series27_weekend_per_weekday [--workers 10]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series25_weekend_training as s25
from src.experiments import run_series26_weekend_features as s26
from src.experiments.series19_21_common import run_parallel, test_days

REPORT_DIR = CONFIG.path("experiments_dir") / "series27_weekend_per_weekday"
OPTIONS = ["none"] + list(s26.VARIANTS)
DAY_TARGET = {"Friday": "log1p", "Saturday": "raw", "Sunday": "log1p"}
SELECTION_BLOCKS = 5  # blocks 0-4 choose, blocks 5-9 judge


def option_predictions(day: str, bags: dict, baseline: dict) -> dict[str, np.ndarray]:
    """Every option's forecasts (all test days) for a weekday, on that weekday's target."""
    target = DAY_TARGET[day]
    base = baseline["log1p_final"] if target == "log1p" else baseline["raw_same_params"]
    return {"none": base, **{v: bags[(v, target)] for v in s26.VARIANTS}}


def choose_options(days: pd.DataFrame, options_by_day: dict[str, dict[str, np.ndarray]], rows: np.ndarray) -> dict[str, str]:
    """Per weekday, the option with the lowest MAE on that weekday's days within `rows`. Ties go to "none"."""
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    chosen = {}
    for day, options in options_by_day.items():
        mask = rows & (wd == day)
        scores = {name: s24.mae(a, pred, mask) for name, pred in options.items()}
        best = min(scores.values())
        chosen[day] = "none" if scores["none"] <= best else min(scores, key=scores.get)
    return chosen


def assemble_weekend(days: pd.DataFrame, hybrid_pred: np.ndarray, options_by_day: dict[str, dict[str, np.ndarray]], chosen: dict[str, str]) -> np.ndarray:
    """The current hybrid, with each weekend day replaced by its chosen option's forecast."""
    wd = days["weekday"].to_numpy()
    out = hybrid_pred.copy()
    for day, option in chosen.items():
        out = np.where(wd == day, options_by_day[day][option], out)
    return out


def table_for(days: pd.DataFrame, rows: np.ndarray, hybrid_pred: np.ndarray, assembled: np.ndarray) -> pd.DataFrame:
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    fs = s25.fri_sun_mask(wd)
    out = []
    for name, mask in [("Friday to Sunday", fs)] + [(d, wd == d) for d in s25.FRI_SUN]:
        m = rows & mask
        out.append({"segment": name, "days": int(m.sum()), "hybrid_MAE": s24.mae(a, hybrid_pred, m), "assembled_MAE": s24.mae(a, assembled, m)})
    t = pd.DataFrame(out)
    t["change"] = t["assembled_MAE"] - t["hybrid_MAE"]
    return t


def evaluation_report(days: pd.DataFrame, rows: np.ndarray, hybrid_pred: np.ndarray, assembled: np.ndarray) -> pd.DataFrame:
    """Change in MAE with 95% block-bootstrap intervals, on the evaluation rows only."""
    sub = days[rows].reset_index(drop=True)
    return s25.winner_report(sub, hybrid_pred[rows], hybrid_pred[rows], assembled[rows], covers_all=False)


def main(workers: int) -> None:
    df, folds = s26.load_frame()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd, fold = days["actual"].to_numpy(), days["weekday"].to_numpy(), days["fold_id"].to_numpy()
    sel, ev = fold < SELECTION_BLOCKS, fold >= SELECTION_BLOCKS
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    b0 = s23.load_bags()
    bags = s26.load_bags(s26.RESULTS, 25, s26.all_variant_targets())
    hybrid0 = s24.hybrid(wd, b0["log1p_final"], b0["raw_same_params"])
    options0 = {d: option_predictions(d, bags, b0) for d in s25.FRI_SUN}

    # ---- step 1: choose on blocks 0-4
    print(f"\nBlocks 0-4 ({int(sel.sum())} days) choose, blocks 5-9 ({int(ev.sum())} days) judge.\n\nStep 1 - MAE by option on the selection blocks (seeds 0-24):")
    sel_table = pd.DataFrame({d: {o: s24.mae(a, p, sel & (wd == d)) for o, p in options0[d].items()} for d in s25.FRI_SUN})
    print(sel_table.round(1).to_string())
    chosen = choose_options(days, options0, sel)
    print("\nChosen on blocks 0-4:", chosen)
    sel_table.to_csv(REPORT_DIR / "selection_blocks0_4.csv")

    # ---- step 2a: judged on blocks 5-9, original seeds
    assembled0 = assemble_weekend(days, hybrid0, options0, chosen)
    print("\nStep 2 - blocks 5-9, seeds 0-24 (the choices never saw these days):\n", table_for(days, ev, hybrid0, assembled0).round(1).to_string(index=False))

    # ---- step 2b: fresh seeds, refitting only the options chosen
    needed = sorted({(chosen[d], DAY_TARGET[d]) for d in chosen if chosen[d] != "none"})
    done_fresh = {(r["variant"], r["target"]) for r in s26.read_done(s26.RESULTS_FRESH).values()}
    todo = [vt for vt in needed if vt not in done_fresh]
    tasks = []
    for variant, target in todo:
        for i in range(0, len(s26.FRESH_SEEDS), s23.SEED_CHUNK):
            chunk = s26.FRESH_SEEDS[i:i + s23.SEED_CHUNK]
            tasks.append({"id": f"{variant}_{target}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "target": target, "seeds": chunk})
    if tasks:
        run_parallel(tasks, s26.fit_task, s26.RESULTS_FRESH, workers, s26.init_worker26, "series27 fresh seeds")
    fresh_bags = s26.load_bags(s26.RESULTS_FRESH, 25, needed) if needed else {}
    f = s24.load_bags(s24.RESULTS_FRESH, 25)
    hybrid_f = s24.hybrid(wd, f["log1p_final"], f["raw_same_params"])
    options_f = {}
    for d in s25.FRI_SUN:
        target = DAY_TARGET[d]
        base = f["log1p_final"] if target == "log1p" else f["raw_same_params"]
        options_f[d] = {"none": base, **{v: fresh_bags[(v, target)] for v in s26.VARIANTS if (v, target) in fresh_bags}}
    assembled_f = assemble_weekend(days, hybrid_f, options_f, chosen)
    rep = evaluation_report(days, ev, hybrid_f, assembled_f)
    rep.to_csv(REPORT_DIR / "evaluation_fresh_seeds.csv", index=False)
    print("\nStep 2 - blocks 5-9, FRESH seeds 100-124 (change = assembled weekend minus the current hybrid, 95% interval):\n", rep.round(1).to_string(index=False))
    ok_a, ok_b, _ = s25.passes_bar(rep)
    print(f"\nBar: (a) Fri-Sun interval below zero: {ok_a}  (b) no weekend day significantly worse: {ok_b}  ->  {'PASSES' if ok_a and ok_b else 'DOES NOT PASS'}")

    # ---- context: choices made on all 585 days (optimistic), and the Series 24 correction on top
    chosen_all = choose_options(days, options0, np.ones(len(a), bool))
    print("\nContext (optimistic): choices made on all 585 days:", chosen_all)
    print(table_for(days, np.ones(len(a), bool), hybrid0, assemble_weekend(days, hybrid0, options0, chosen_all)).round(1).to_string(index=False))
    sc = fold >= 1
    for name, base in (("current hybrid + Series 24 correction", hybrid_f), ("chosen weekend + Series 24 correction", assembled_f)):
        c = s24.correct(days, base, 8, 0.5, scope=("Saturday", "Sunday"))
        print(f"Context: {name:42s} MAE on blocks 5-9 {s24.mae(a, c, ev):.2f}   (uncorrected {s24.mae(a, base, ev):.2f})   scored blocks 1-9 {s24.mae(a, c, sc):.2f}")
    if any(o.startswith("A_") or o == "AB" for o in chosen.values()):
        print("Serving caveat: a chosen option uses recent sales, which the API does not have weeks ahead.")
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args().workers)
