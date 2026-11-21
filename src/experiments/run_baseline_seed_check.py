"""
Baseline seed check - is the hydro Monday-to-Thursday and Saturday-fallback pattern of Series 54-56 a lucky shared baseline?

Across Series 54-56 (12 calendar variants, each added to every piece of the weekly log and compared with the same "none" fit of the piece):

  hydro Monday-Thursday   10 of 12 variants beat today's piece (mean -0.96 pounds a day), but the plain Monday-Thursday piece, which differs from it
                          only by the `hydro_thu` flag, beats it in 1 of 12 (mean +1.54)
  Saturday fallback (raw) 10 of 12 variants beat today's piece in the backtest (mean -2.83) and 10 of 12 are WORSE on the 56 new days

Every variant of a piece is compared with that piece's single "none" bag of 100 seeds (200-299), so if that bag happened to land a little high, every
variant would look better against it. That would also fit the new days landing the other way. This check refits the "none" baselines of those two
pieces, and of the plain Monday-Thursday piece as a control, on 100 fresh seeds (300-399) with the same columns, folds and data, and asks how far a
baseline moves with the seeds and what the 12 variants look like against a better baseline.

  none (fresh seeds)  the three pieces as deployed, seeds 300-399, 585 backtest days and the 56 new days

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds, and the 56 days from 7 September 2026; the variants are Series 54-56's own
             fits (seeds 200-299, unchanged); only the baselines are refitted
  look at    each piece's baseline MAE on its own weekdays under seeds 200-299 and under 300-399, their difference with a 95% moving-block bootstrap
             interval and in how many folds the fresh one is better; then the 12 variants' changes against the old baseline, the fresh one and the
             average of the two (a 200-seed baseline): how many beat it and the mean change, on the backtest and on the new days
  expected   the two 100-seed baselines of a piece differ by about 1 to 2 pounds a day on its own weekdays (Series 44-46 saw 2.8 between two
             25-seed bags of the whole model), which is as large as the hydro piece's mean gain; so I expect the hydro pattern to shrink a lot
             against the average baseline and the control to move too
  decision   the pattern is mostly baseline luck if, against the averaged baseline, the hydro piece keeps less than half its mean gain
             (mean above -0.5) or fewer than 8 of its 12 variants still beat it; real if it keeps most of it (mean below -0.75, 8 or more of 12)
  limits     no model changes; nothing in production changes; the post model's four baselines are not refitted

Run:  python -m src.experiments.run_baseline_seed_check list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series55_week_of_month as s55
from src.experiments import run_series56_week_of_year_and_month_edges as s56
from src.experiments.series19_21_common import DATE_COL, TARGET_COL, load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "baseline_seed_check"
RESULTS = OUT_DIR / "baseline_seed_check.jsonl"
FRESH_SEEDS = list(range(300, 400))
PIECES = ["Monday-Thursday", "Monday-Thursday (hydro)", "Saturday fallback (raw)"]
CUTOFF = pd.Timestamp("2026-09-07")


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        features = s54.piece_features(piece, "none")
        for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
            for i in range(0, len(FRESH_SEEDS), size):
                chunk = FRESH_SEEDS[i:i + size]
                out.append({"id": f"{part}_{piece}_none_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": "none",
                            "seeds": chunk, "features": features})
    return out


def fresh_bag(part: str, piece: str) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in read_done(RESULTS).values() if (r["part"], r["piece"]) == (part, piece)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(FRESH_SEEDS), (part, piece, per_seed.shape)
    return per_seed.mean(axis=0)


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    old_bt, old_nd = s54.load_bags("backtest", s54.RESULTS), s54.load_bags("new_days", s54.RESULTS)
    series = {54: (s54.RESULTS, s54.VARIANT_COLUMNS), 55: (s55.RESULTS, s55.VARIANT_COLUMNS), 56: (s56.RESULTS, s56.VARIANT_COLUMNS)}
    variants = {s: (s54.load_bags("backtest", res, cols, s54.RESULTS), s54.load_bags("new_days", res, cols, s54.RESULTS)) for s, (res, cols) in series.items()}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("THE BASELINE ON ITS OWN WEEKDAYS: seeds 200-299 against seeds 300-399 (difference = fresh minus old; negative = the fresh bag is better)")
    rows = []
    for piece in PIECES:
        weekdays = s54.PIECES[piece][2]
        own, own_new = np.isin(dow, weekdays), np.isin(dow_new, weekdays)
        f_bt, f_nd = fresh_bag("backtest", piece), fresh_bag("new_days", piece)
        o_bt, o_nd = old_bt[(piece, "none")], old_nd[(piece, "none")]
        d = np.abs(f_bt - a) - np.abs(o_bt - a)
        change, lo, hi = s23.masked_block_bootstrap(d, own.astype(float))
        rows.append({"piece": piece, "MAE_200_299": float(np.abs(o_bt - a)[own].mean()), "MAE_300_399": float(np.abs(f_bt - a)[own].mean()),
                     "difference": change, "lo": lo, "hi": hi, "folds_fresh_better": int(sum(d[own & (fold == k)].mean() < 0 for k in range(10))),
                     "new_days_MAE_200_299": float(np.abs(o_nd - a_new)[own_new].mean()), "new_days_MAE_300_399": float(np.abs(f_nd - a_new)[own_new].mean())})
    base = pd.DataFrame(rows)
    base.to_csv(OUT_DIR / "baselines.csv", index=False)
    print(base.round(2).to_string(index=False))

    print("\n\nTHE 12 CALENDAR VARIANTS OF SERIES 54-56 AGAINST EACH BASELINE (change on the piece's own weekdays, pounds a day; negative = better)")
    out = []
    for piece in PIECES:
        weekdays = s54.PIECES[piece][2]
        own, own_new = np.isin(dow, weekdays), np.isin(dow_new, weekdays)
        f_bt, f_nd = fresh_bag("backtest", piece), fresh_bag("new_days", piece)
        o_bt, o_nd = old_bt[(piece, "none")], old_nd[(piece, "none")]
        baselines = {"old (200-299)": (o_bt, o_nd), "fresh (300-399)": (f_bt, f_nd), "average of both": ((o_bt + f_bt) / 2, (o_nd + f_nd) / 2)}
        for s, (bt, nd) in variants.items():
            for v in [v for v in series[s][1] if v != "none"]:
                row = {"piece": piece, "series": s, "variant": v}
                for name, (b_bt, b_nd) in baselines.items():
                    row[f"backtest vs {name}"] = float((np.abs(bt[(piece, v)] - a) - np.abs(b_bt - a))[own].mean())
                    row[f"new days vs {name}"] = float((np.abs(nd[(piece, v)] - a_new) - np.abs(b_nd - a_new))[own_new].mean())
                out.append(row)
    table = pd.DataFrame(out)
    table.to_csv(OUT_DIR / "variants_against_baselines.csv", index=False)
    summary = []
    for piece in PIECES:
        t = table[table["piece"] == piece]
        for part in ("backtest", "new days"):
            line = {"piece": piece, "part": part}
            for name in ("old (200-299)", "fresh (300-399)", "average of both"):
                col = f"{part} vs {name}"
                line[f"{name}: beat/12"] = int((t[col] < 0).sum())
                line[f"{name}: mean"] = round(float(t[col].mean()), 2)
            summary.append(line)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PIECES)} pieces x seeds {FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]}, {len(tasks)} tasks")
        for piece in PIECES:
            print(f"  {piece:26s} {len(s54.piece_features(piece, 'none'))} columns, weekdays {s54.PIECES[piece][2]}")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, s54.init_worker54, "baseline_seed_check")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
