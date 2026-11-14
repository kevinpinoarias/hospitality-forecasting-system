"""
Series 92 - the year-end candidates on the rest of the seeds: is the specialist windows' lead over the simpler alternatives real?

Series 90 (25 seeds, 200-224, served with the S6 stand-in) ranked the year-end candidates: the specialist windows M1 (the 1 to 8 January masked model on 1-8 January, the companion without the history inputs on 26-31 December, the
unmasked model elsewhere) -10.11 pounds a day; the combinations built afterwards from stored forecasts D + the companion on 26-31 December -8.62 and B + the companion -8.83; D alone (the ten history inputs rebuilt without
24 December-2 January) -4.76 and B alone (the sales on the same weekday 364 days earlier as an input, `sales_364`) -3.02. The gaps between M1 and the combinations (1.3 to 1.5) are inside 25-seed noise (about 2). Kevin
(9 November 2026): repeat on the other seeds to see whether the difference is only noise.

This series refits D and B (the four pieces of the current package, as Series 90) on:
  seeds 225-299   the other 75 of the old 100-seed set (with Series 90's 25 they have all 100)
  seeds 300-399   fresh seeds
both served with the S6 stand-in (the hybrid of Series 89; models fitted on the real training rows, only the test rows' `forecast_sales` differs, history from the training rows only). The references are Series 89's S6-served
fits on the same seeds: the unmasked package, the 1 to 8 January masked model and the companion (all 100 seeds and the fresh seeds). 2 candidates x 4 pieces x (15 + 20) tasks of 5 seeds = 280 tasks.

The compared packages (each against the unmasked package served with S6, after the deployed weekend correction):
  M1        the specialist windows
  D         D in every piece
  D+comp    D in every piece, with the companion's forecast on 26-31 December
  D+mask+comp   as D+comp, with the masked model's forecast on 1-8 January too
  B         B in every piece
  B+comp    B in every piece, with the companion's forecast on 26-31 December

READING (set before any result was seen)
  measure     the net change of the package over the 585 days against the unmasked package served with S6 (pounds a day; negative = better), on (a) all 100 seeds and (b) the fresh seeds; 26-31 Dec, 1-8 Jan and all
              other days shown
  M1 ahead    M1 is better than the candidate by more than 2.0 on BOTH seed sets;  tied: within 2.0 on both;  candidate ahead: better than M1 by more than 2.0 on both;  mixed otherwise
  also        the 25-seed figures of Series 90 are shown beside them
  limits      the same 585 days and the same one December and one January, so this checks the seeds and not the days; the combinations are not chosen by a rule fixed before the first look (they were built after
              Series 90); a different stand-in (S6) changes an input of the pre-registered log and any model change is a new forecast-log version, both Kevin's decisions and both after an amendment

Run:  python -m src.experiments.run_series92_year_end_seeds_repeat list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series89_standin_repeat_hybrid as s89
from src.experiments import run_series90_year_end_candidates as s90
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series92"
RESULTS = OUT_DIR / "series92_year_end_seeds_repeat.jsonl"
PIECES = s90.PIECES
CANDIDATES = ["D", "B"]
REST, FRESH = SEEDS[25:], list(range(300, 400))


def make_tasks() -> list[dict]:
    out = []
    for c in CANDIDATES:
        for piece in PIECES:
            features = s90.features_for(c, piece)
            for name, seeds in ((c, REST), (f"{c}_fresh", FRESH)):
                for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
                    chunk = seeds[i:i + s54.BACKTEST_CHUNK]
                    out.append({"id": f"backtest_{piece}_{name}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": name, "seeds": chunk, "features": features})
    return out


def collect(done: dict, piece: str, variant: str, n: int, key: str = "per_seed") -> np.ndarray:
    rows = [np.array(r[key] if key == "per_seed" else r[key]["S6"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
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
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    jan8 = s82.in_window(days["date"], (1, 1), (1, 8))
    done92, done90, done89 = read_done(RESULTS), read_done(s90.RESULTS), read_done(s89.RESULTS)
    pd.set_option("display.width", 250)

    def package(p: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def s89p(variant: str, suffix: str) -> dict:
        return {pc: s89.mine(done89, pc, variant + suffix, "S6") for pc in PIECES}

    def cand(c: str, suffix: str) -> dict:
        if suffix == "":  # all 100 seeds: Series 90's 25 plus the new 75
            return {pc: (s90.mine(done90, pc, c) * 25 + collect(done92, pc, c, 75) * 75) / 100 for pc in PIECES}
        return {pc: collect(done92, pc, f"{c}_fresh", 100) for pc in PIECES}

    tables, nets = {}, {}
    for label, suffix in (("all 100 seeds", ""), ("fresh seeds 300-399", "_fresh")):
        base, msk, comp = s89p("sd_base", suffix), s89p("sd_j1_8", suffix), s89p("sd_companion", suffix)
        ref = package(base)
        D, B = cand("D", suffix), cand("B", suffix)
        swap = lambda main, mask, other: {pc: np.where(mask, other[pc], main[pc]) for pc in PIECES}  # noqa: E731
        variants = {"M1": package(swap(swap(base, jan8, msk), six, comp)), "D": package(D), "D+comp": package(swap(D, six, comp)), "D+mask+comp": package(swap(swap(D, jan8, msk), six, comp)),
                    "B": package(B), "B+comp": package(swap(B, six, comp))}
        rows = {}
        for name, p in variants.items():
            d = np.abs(p - a) - np.abs(ref - a)
            rows[name] = {"net, 585 days": float(d.mean()), "26-31 Dec": float(d[six].mean()), "1-8 Jan": float(d[jan8].mean()), "all other days": float(d[~(six | jan8)].mean())}
        tables[label] = pd.DataFrame(rows).T
        nets[label] = tables[label]["net, 585 days"]
        print(f"{label.upper()}: CHANGE AGAINST THE UNMASKED PACKAGE, all served with the S6 stand-in (pounds a day; negative = better)\n")
        print(tables[label].round(2).to_string())
        print()
    s25 = {"M1": -10.11, "D": -4.76, "D+comp": -8.62, "D+mask+comp": -8.77, "B": -3.02, "B+comp": -8.83}
    print("SERIES 90's 25-SEED FIGURES, for comparison:", s25, "\n")
    for c in ("D", "D+comp", "D+mask+comp", "B", "B+comp"):
        gaps = {k: float(nets[k][c] - nets[k]["M1"]) for k in nets}  # positive = M1 better
        if all(g > 2.0 for g in gaps.values()):
            verdict = "M1 AHEAD (better by more than 2.0 on both seed sets)"
        elif all(g < -2.0 for g in gaps.values()):
            verdict = "THE CANDIDATE IS AHEAD (better by more than 2.0 on both)"
        elif all(abs(g) <= 2.0 for g in gaps.values()):
            verdict = "TIED (within 2.0 on both)"
        else:
            verdict = "MIXED"
        print(f"READING SET BEFOREHAND, {c} against M1: " + ", ".join(f"{k}: {g:+.2f}" for k, g in gaps.items()) + f"  ->  {verdict}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(OUT_DIR / f"packages_{'fresh' if 'fresh' in k else 'seeds100'}.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(CANDIDATES)} candidates x {len(PIECES)} pieces x (15 tasks on seeds 225-299 + 20 on the fresh seeds) = {len(tasks)} tasks (backtest only, served with S6)")
        d90 = read_done(s90.RESULTS)
        d89 = read_done(s89.RESULTS)
        for pc in PIECES:
            for c in CANDIDATES:
                s90.mine(d90, pc, c)
            for v in ("sd_base", "sd_j1_8", "sd_companion"):
                s89.mine(d89, pc, v, "S6")
                s89.mine(d89, pc, v + "_fresh", "S6")
        print("Series 90's 25-seed fits and Series 89's S6-served references (100 seeds and fresh) are present for every piece")
    elif step == "run":
        run_parallel(make_tasks(), s90.fit_task90, RESULTS, workers, s90.init_worker90, "series92")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
