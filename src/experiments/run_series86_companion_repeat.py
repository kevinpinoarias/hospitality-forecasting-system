"""
Series 86 - the companion model (Series 85) on the rest of the seeds.

Series 85 (25 seeds, 200-224): the 1 to 8 January mask plus a companion model (each piece refitted without the ten history inputs) used only on 26 to 31 December reached net -11.30 pounds a day over the 585 days, against
-9.07 for the mask alone (-2.22), with the six days at -301 against -64 and the other days keeping their gain (-4.02 against -4.23). The six days are one December. Kevin asked (9 November 2026) for the seeds to be repeated:
  seeds 225-299   the other 75 of the old 100-seed set (the companion refitted on them; with Series 85's 25 it has all 100); compared with the existing 100-seed baseline (Series 80's unmasked package, Saturday on Series 77's
                  closed-day column) and with the 1 to 8 January mask on all 100 seeds (Series 82's 25 plus Series 83's 75)
  seeds 300-399   fresh seeds; compared with the unmasked package on the same seeds (Series 80's `base_fresh`, Series 77's `xmas_only_fresh`) and the 1 to 8 January mask on them (Series 83's `j1_8_fresh`)
The companion is exactly Series 85's (same inputs, pieces and folds); the variants are built the same way, swapping its forecast in on 26 to 31 December before the weekend correction.
4 pieces x (15 + 20) tasks of 5 seeds = 140 tasks.

READING (set before any result was seen)
  measure      the NET change of the post package over the 585 days against the unmasked package (pounds a day; negative = better) for `comp_dec` (the mask + the companion on 26-31 December), beside the 1 to 8
               January mask alone, on (a) all 100 seeds of the old set and (b) the fresh seeds 300-399; the six days' change and the other days' change are shown
  replicates   comp_dec is better than the mask alone by at least 1.1 (half of Series 85's 2.22) on BOTH the 100 seeds and the fresh seeds
  weakens      better than the mask alone on both, but under 1.1 on one of them
  fails        not better than the mask alone on one of them
  limits       the same 585 days and the same one December (six days), so this checks the seeds, not the days; the companion leans on the manager's number for the day (Series 87 tests it without that number); a refit
               would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series86_companion_repeat list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series70_payday_intensity as s70
from src.experiments import run_series76_december_early_pay as s76
from src.experiments import run_series77_christmas_closed_days as s77
from src.experiments import run_series80_january_lags_masked as s80
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series83_mask_windows_repeat as s83
from src.experiments import run_series85_companion_model as s85
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series86"
RESULTS = OUT_DIR / "series86_companion_repeat.jsonl"
PIECES = s85.PIECES
REST, FRESH = SEEDS[25:], list(range(300, 400))
FIRST_25_GAIN = 2.22


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        features = s85.piece_features(piece)
        for name, seeds in (("companion", REST), ("companion_fresh", FRESH)):
            for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
                chunk = seeds[i:i + s54.BACKTEST_CHUNK]
                out.append({"id": f"backtest_{piece}_{name}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": name, "seeds": chunk, "features": features})
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    assert six.sum() == 6
    done86, done85, done82, done83 = read_done(RESULTS), read_done(s85.RESULTS), read_done(s82.RESULTS), read_done(s83.RESULTS)
    done80, done77 = read_done(s80.RESULTS), read_done(s77.RESULTS)
    bags = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    pd.set_option("display.width", 250)

    def package(pieces: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [pieces["Friday"], pieces["Saturday"], pieces["Sunday"]], default=pieces["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def blend(srcs: list[dict], pc: str, variant: str, counts: list[int]) -> np.ndarray:
        return sum(s83.collect([s], pc, variant, n) * n for s, n in zip(srcs, counts)) / sum(counts)

    base100 = {"Monday-Thursday (hydro)": bags[("Monday-Thursday (hydro)", "none")], "Friday": bags[("Friday", "none")], "Saturday": s76.mine(done77, "xmas_only", SEEDS), "Sunday": bags[("Sunday", "none")]}
    base_f = {"Monday-Thursday (hydro)": s80.mine(done80, "Monday-Thursday (hydro)", "base_fresh", FRESH), "Friday": s80.mine(done80, "Friday", "base_fresh", FRESH),
              "Saturday": s76.mine(done77, "xmas_only_fresh", FRESH), "Sunday": s80.mine(done80, "Sunday", "base_fresh", FRESH)}
    jan100 = {pc: blend([done82, done83], pc, "j1_8", [25, 75]) for pc in PIECES}
    jan_f = {pc: s83.collect([done83], pc, "j1_8_fresh", 100) for pc in PIECES}
    comp100 = {pc: blend([done85, done86], pc, "companion", [25, 75]) for pc in PIECES}
    comp_f = {pc: s83.collect([done86], pc, "companion_fresh", 100) for pc in PIECES}

    def swap(main: dict, comp: dict) -> dict:
        return {pc: np.where(six, comp[pc], main[pc]) for pc in PIECES}

    rows = {}
    results = {}
    for label, base, jan, comp in (("all 100 seeds", base100, jan100, comp100), ("fresh seeds 300-399", base_f, jan_f, comp_f)):
        b = package(base)
        for name, pred in (("1 to 8 Jan mask alone", package(jan)), ("mask + companion on 26-31 Dec", package(swap(jan, comp)))):
            diff = np.abs(pred - a) - np.abs(b - a)
            rows[(label, name)] = {"net, 585 days": float(diff.mean()), "the 6 days 26-31 Dec": float(diff[six].mean()), "all other days": float(diff[~six].mean())}
        results[label] = rows[(label, "mask + companion on 26-31 Dec")]["net, 585 days"] - rows[(label, "1 to 8 Jan mask alone")]["net, 585 days"]
    table = pd.DataFrame(rows).T
    print("EACH VARIANT MINUS THE UNMASKED PACKAGE (pounds a day; negative = better)\n")
    print(table.round(2).to_string())
    print(f"\ncomp_dec against the mask alone: {results['all 100 seeds']:+.2f} on the 100 seeds and {results['fresh seeds 300-399']:+.2f} on the fresh seeds (Series 85, 25 seeds: -2.22)")
    gains = [-results["all 100 seeds"], -results["fresh seeds 300-399"]]
    verdict = "REPLICATES" if all(g >= FIRST_25_GAIN / 2 for g in gains) else "WEAKENS" if all(g > 0 for g in gains) else "FAILS"
    print(f"READING SET BEFOREHAND: {verdict}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "companion_repeat.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"the companion for {len(PIECES)} pieces x (15 tasks on seeds 225-299 + 20 on the fresh seeds) = {len(tasks)} tasks (backtest only)")
        for pc in PIECES:
            s83.collect([read_done(s85.RESULTS)], pc, "companion", 25)
            s83.collect([read_done(s83.RESULTS)], pc, "j1_8_fresh", 100)
        print("Series 85's 25-seed companion fits and Series 83's fresh-seed 1-to-8-January fits are present for every piece")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, s85.init_worker85, "series86")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
