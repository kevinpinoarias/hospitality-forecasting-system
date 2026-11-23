"""
Series 85 - a companion model without the history inputs for the late-December days (option 2b of the January working list).

Series 83 and 84: switching the ten recent-sales inputs off for 1 to 8 January gains about 8 to 10 pounds a day over the 585 days, and the six late-December days (26 to 31 December) would gain about 460 pounds a day each from a
mask too, but masking those rows in TRAINING costs the gain on every other day (the recent averages and lags in late December are good lessons), and masking only some inputs there captures little (Series 84).
Option 2b (Kevin, 9 November 2026): a COMPANION model. Each piece is refitted WITHOUT the ten history inputs on all rows, from the calendar, the weather, the hydro flag, the closed-day intensity (Saturday) and the manager's
number (`forecast_sales`); its forecast is used only on chosen days, and the main model keeps its normal training rows, so no training information is lost. No missing values are involved (a tree trained without them reads a
missing value as the lowest value it knows and forecasts low).
The ten inputs: `lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`, `lag_1_fe`, `lag_7_fe`, `rolling_7_fe`, `lag_14_sales`, `lag_21_sales`, `same_wd_mean4`, `same_wd_max4` (a piece only has the ones it uses).

One new fit: the companion for the four pieces of the current package (Monday-Thursday with the hydro flag, Friday, Saturday with `intensity_fixed`, Sunday), 25 seeds (200-224), 10 folds, 585 days: 4 x 5 tasks = 20. The
variants are built from stored forecasts (Series 81's `base`, Series 82's `j1_8`), swapping the companion's forecast in on chosen days before the deployed weekend correction:
  comp_dec        the main model is `j1_8` (the 1 to 8 January mask, trained and used as in Series 82/83); the companion replaces it on 26 to 31 December
  comp_dec_jan    the main model is `j1_8`; the companion replaces it on 26 December to 8 January (the 1 to 8 January days get the companion's number instead of the masked main model's)
  comp_only       the main model is the UNMASKED `base`; the companion replaces it on 26 December to 8 January (no masked training rows: shows what the companion does on its own)

READING (set before any result was seen)
  measure     the NET change of the post package over the 585 days against `base` (pounds a day; negative = better); also the change over the six days 26 to 31 December 2025 (the only late-December days in the backtest)
              and over all other days
  captures    comp_dec captures BOTH when its net change is at least 2.0 better than j1_8's (25-seed noise is about 2): since it differs from j1_8 only on the six days, that is a gain of at least about 195 pounds a day
              on those six days beyond what the masked main model already gives (j1_8 leaves them unmasked)
  no gain     within 2.0 of j1_8's net;  costs  worse than j1_8's by more than 2.0
  also        comp_dec_jan and comp_only are judged the same way against j1_8 and reported as they are
  expected    modest: the companion has no recent level, so on a day where the recent level matters it will be worse than the main model; late December is the case where the level misleads, so a gain there is
              plausible, but the six days are one December and the manager's number (a real input in the backtest, a median stand-in in the live log) does much of the work
  limits      one December in the backtest (six days), so the late-December result rests on six days; 25 seeds; in the live log `forecast_sales` is a stand-in, which weakens every piece that leans on it, the companion
              most; the same 585 days; a refit would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series85_companion_model list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series81_closed_days_in_lags as s81
from src.experiments import run_series82_mask_windows as s82
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel, test_days, read_done
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series85"
RESULTS = OUT_DIR / "series85_companion_model.jsonl"
SEEDS25, PIECES, INPUTS = s81.SEEDS25, s81.PIECES, s81.INPUTS
VARIANT = "companion"


def piece_features(piece: str) -> list[str]:
    features = s54.piece_features(piece, "none") + ([pi.COLUMN_FIXED] if piece == "Saturday" else [])
    return [f for f in features if f not in INPUTS]


def init_worker85() -> None:
    s47.init_worker47()
    _STATE["df"] = pi.add_payday_intensity_fixed(add_hydro_thu(_STATE["df"]), DATE_COL)


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        features = piece_features(piece)
        for i in range(0, len(SEEDS25), s54.BACKTEST_CHUNK):
            chunk = SEEDS25[i:i + s54.BACKTEST_CHUNK]
            out.append({"id": f"backtest_{piece}_{VARIANT}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": VARIANT, "seeds": chunk, "features": features})
    return out


def mine(done: dict, piece: str, variant: str) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(SEEDS25), (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    fourteen = s82.in_window(days["date"], (12, 26), (1, 8))
    assert six.sum() == 6 and fourteen.sum() == 14, (int(six.sum()), int(fourteen.sum()))
    done, done81, done82 = read_done(RESULTS), read_done(s81.RESULTS), read_done(s82.RESULTS)
    pd.set_option("display.width", 250)

    def per_piece(source: dict, variant: str) -> dict:
        return {name: mine(source, name, variant) for name in PIECES}

    base_p, jan_p, comp_p = per_piece(done81, "base"), per_piece(done82, "j1_8"), per_piece(done, VARIANT)

    def package(pieces: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [pieces["Friday"], pieces["Saturday"], pieces["Sunday"]], default=pieces["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def swapped(main: dict, mask: np.ndarray) -> dict:
        return {name: np.where(mask, comp_p[name], main[name]) for name in PIECES}

    base = package(base_p)
    variants = {"j1_8 (the reference)": package(jan_p), "comp_dec": package(swapped(jan_p, six)), "comp_dec_jan": package(swapped(jan_p, fourteen)), "comp_only": package(swapped(base_p, fourteen)),
                "the companion on every day": package(comp_p)}
    rows = {}
    for name, p in variants.items():
        diff = np.abs(p - a) - np.abs(base - a)
        rows[name] = {"net, 585 days": float(diff.mean()), "the 6 days 26-31 Dec": float(diff[six].mean()), "the 14 days 26 Dec-8 Jan": float(diff[fourteen].mean()), "all other days": float(diff[~fourteen].mean())}
    table = pd.DataFrame(rows).T
    print("EACH VARIANT MINUS THE UNMASKED PACKAGE, the same 25 seeds (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    ref = float(table.loc["j1_8 (the reference)", "net, 585 days"])
    print(f"\nj1_8: net {ref:+.2f}")
    for v in ("comp_dec", "comp_dec_jan", "comp_only"):
        net = float(table.loc[v, "net, 585 days"])
        verdict = "CAPTURES BOTH (2.0 or more better than j1_8)" if net <= ref - 2.0 else "COSTS (more than 2.0 worse than j1_8)" if net > ref + 2.0 else "NO GAIN (within 2.0 of j1_8)"
        print(f"READING SET BEFOREHAND, {v}: net {net:+.2f} ({net - ref:+.2f} against j1_8): {verdict}")
    # what the companion does on its own across the whole backtest: the price of having no history inputs
    ratio = np.abs(package(comp_p) - a).mean() / np.abs(base - a).mean()
    print(f"\nthe companion alone, on every day, has an average miss {np.abs(package(comp_p) - a).mean():,.0f} against {np.abs(base - a).mean():,.0f} for the unmasked package ({(ratio - 1) * 100:+.0f}%)")
    detail = days.loc[fourteen, ["date", "weekday", "actual"]].copy()
    detail["unmasked"], detail["j1_8"], detail["companion"] = base[fourteen], package(jan_p)[fourteen], package(comp_p)[fourteen]
    detail["unmasked - actual"], detail["j1_8 - actual"], detail["companion - actual"] = detail["unmasked"] - detail["actual"], detail["j1_8"] - detail["actual"], detail["companion"] - detail["actual"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "variants.csv")
    detail.round(0).to_csv(OUT_DIR / "days_26dec_8jan.csv", index=False)
    print("\nTHE PACKAGE'S FORECAST ON 26 DECEMBER 2025 TO 8 JANUARY 2026:\n")
    print(detail.round(0).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"the companion for {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds)")
        for piece in PIECES:
            f = piece_features(piece)
            dropped = [x for x in s54.piece_features(piece, "none") if x in INPUTS]
            assert not (set(f) & set(INPUTS)), f"{piece}: a history input is left in"
            print(f"{piece:26s} {len(f)} inputs (dropped {len(dropped)} history inputs: {dropped})")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker85, "series85")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
