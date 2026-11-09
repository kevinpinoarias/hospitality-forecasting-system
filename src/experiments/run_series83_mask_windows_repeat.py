"""
Series 83 - the two best mask windows of Series 82 on the rest of the seeds.

Series 82 (25 seeds, 200-224) found two windows for switching the ten recent-sales inputs off around the new year within 1.0 pounds a day of each other: 1 to 8 January (-9.07 over the 585 days) and 26 December to 8 January
(-8.34), against -6.08 for 1 to 15 January; a difference of 2 or less is within 25-seed noise, and 1 to 8 January against 1 to 15 January (3.0) was only just beyond it. Kevin asked (9 November 2026) for these two to be repeated:
  seeds 225-299   the other 75 seeds of the old 100-seed set, so each window gets all 100 seeds (200-299) and can be compared with the existing 100-seed baseline (Series 80's: the unmasked package, Saturday with the closed-day
                  column = Series 77's xmas_only)
  seeds 300-399   fresh seeds, compared with the unmasked package on the same seeds (Series 80's `base_fresh` and Series 77's `xmas_only_fresh`)
Same masks, pieces and folds as Series 82 (all four pieces of the current package, the ten inputs missing on every row inside the window, training rows and forecast days alike, 585 days, the deployed weekend correction).
2 windows x 4 pieces x (15 + 20) tasks of 5 seeds = 280 tasks.

READING (set before any result was seen)
  measure      the NET change of the post package over the 585 days (pounds a day; negative = better) on (a) all 100 seeds of the old set, (b) the fresh seeds 300-399; the 25-seed figure is shown beside them
  replicates   both the 100-seed and the fresh-seed net change are negative and each at least half the size of the 25-seed result (4.5 for 1 to 8 January, 4.2 for 26 December to 8 January)
  weakens      both negative but one of them under half the 25-seed size
  fails        either is zero or positive
  which window the two windows are TIED if their net changes are within 1.0 on both the 100 seeds and the fresh seeds; otherwise the better one on the average of the two is reported as ahead
  also         the change inside the window's own days is shown (8 test days for 1 to 8 January, 14 for 26 December to 8 January)
  limits       the same 585 days throughout, so this checks the seeds, not the days; the window is still the best of six tries; one January in the backtest; a rule written down beforehand and read on January 2027 is
               the real test; a refit would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series83_mask_windows_repeat list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series70_payday_intensity as s70
from src.experiments import run_series76_december_early_pay as s76
from src.experiments import run_series77_christmas_closed_days as s77
from src.experiments import run_series80_january_lags_masked as s80
from src.experiments import run_series82_mask_windows as s82
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series83"
RESULTS = OUT_DIR / "series83_mask_windows_repeat.jsonl"
VARIANTS = ["j1_8", "d26_j8"]
PIECES = s82.PIECES
REST = SEEDS[25:]  # seeds 225-299
FRESH = list(range(300, 400))
FIRST_25 = {"j1_8": -9.07, "d26_j8": -8.34}


def make_tasks() -> list[dict]:
    out = []
    for variant in VARIANTS:
        for piece in PIECES:
            features = s82.piece_features(piece, variant)
            for name, seeds in ((variant, REST), (f"{variant}_fresh", FRESH)):
                for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
                    chunk = seeds[i:i + s54.BACKTEST_CHUNK]
                    out.append({"id": f"backtest_{piece}_{name}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": name, "seeds": chunk, "features": features})
    return out


def collect(sources: list[dict], piece: str, variant: str, n: int) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for src in sources for r in src.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == n, (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    done, done82, done80, done77 = read_done(RESULTS), read_done(s82.RESULTS), read_done(s80.RESULTS), read_done(s77.RESULTS)
    bags = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    pd.set_option("display.width", 250)

    def package(mon_thu, fri, sat, sun):
        pred = np.select([dow == 4, dow == 5, dow == 6], [fri, sat, sun], default=mon_thu)
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base100 = package(bags[("Monday-Thursday (hydro)", "none")], bags[("Friday", "none")], s76.mine(done77, "xmas_only", SEEDS), bags[("Sunday", "none")])
    base_fresh = package(s80.mine(done80, "Monday-Thursday (hydro)", "base_fresh", FRESH), s80.mine(done80, "Friday", "base_fresh", FRESH), s76.mine(done77, "xmas_only_fresh", FRESH),
                         s80.mine(done80, "Sunday", "base_fresh", FRESH))
    rows, verdicts = {}, {}
    for variant in VARIANTS:
        first, last = s82.WINDOWS[variant][0], s82.WINDOWS[variant][1]
        inside = s82.in_window(days["date"], first, last)
        # all 100 seeds of the old set: Series 82's 25 plus the other 75
        p100 = package(*((collect([done82], pc, variant, 25) * 25 + collect([done], pc, variant, 75) * 75) / 100 for pc in PIECES))
        pf = package(*(collect([done], p, f"{variant}_fresh", 100) for p in PIECES))
        d100, df_ = np.abs(p100 - a) - np.abs(base100 - a), np.abs(pf - a) - np.abs(base_fresh - a)
        rows[variant] = {"25 seeds (Series 82)": FIRST_25[variant], "all 100 seeds, net": float(d100.mean()), "fresh 300-399, net": float(df_.mean()),
                         "own days, 100 seeds": float(d100[inside].mean()), "own days, fresh": float(df_[inside].mean()), "outside, 100 seeds": float(d100[~inside].mean()), "outside, fresh": float(df_[~inside].mean())}
        half = abs(FIRST_25[variant]) / 2
        n100, nf = rows[variant]["all 100 seeds, net"], rows[variant]["fresh 300-399, net"]
        verdicts[variant] = "REPLICATES" if (n100 < 0 and nf < 0 and abs(n100) >= half and abs(nf) >= half) else "WEAKENS" if (n100 < 0 and nf < 0) else "FAILS"
    table = pd.DataFrame(rows).T
    print("EACH WINDOW MINUS THE UNMASKED PACKAGE (pounds a day; negative = better)\n")
    print(table.round(2).to_string())
    for v in VARIANTS:
        print(f"READING SET BEFOREHAND, {v}: {verdicts[v]}")
    d_100 = abs(table.loc["j1_8", "all 100 seeds, net"] - table.loc["d26_j8", "all 100 seeds, net"])
    d_f = abs(table.loc["j1_8", "fresh 300-399, net"] - table.loc["d26_j8", "fresh 300-399, net"])
    avg = table[["all 100 seeds, net", "fresh 300-399, net"]].mean(axis=1)
    print(f"\nthe two windows differ by {d_100:.2f} on the 100 seeds and {d_f:.2f} on the fresh seeds: {'TIED' if d_100 <= 1.0 and d_f <= 1.0 else 'not tied, ahead on the average of the two: ' + avg.idxmin()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "windows_repeat.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} windows x {len(PIECES)} pieces x (15 tasks on seeds 225-299 + 20 on the fresh seeds) = {len(tasks)} tasks (backtest only)")
        done82 = read_done(s82.RESULTS)
        for v in VARIANTS:
            for pc in PIECES:
                collect([done82], pc, v, 25)  # asserts that all 25 seeds are there
        print("Series 82's 25-seed fits for both windows are present for every piece")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, s82.init_worker82, "series83")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
