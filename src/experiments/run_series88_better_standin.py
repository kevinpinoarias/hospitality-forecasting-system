"""
Series 88 - a better stand-in for the manager's number around the holidays.

The live forecast log cannot use the manager's real forecast for the day (it reaches us after the day), so every log forecast is served with a STAND-IN: the weekday-and-season median of the manager's past forecasts
(`seasonal_medians`: the same weekday within 10 days of the same day of the year, from the history). Series 87 showed that serving this stand-in costs the unmasked package 91 pounds a day (average miss 945 against 854 with
the real number; Series 43: about 99), that the January and late-December gains of Series 80-86 mostly vanish with it, and that the stand-in is weak around the holidays: with two years of history the median is contaminated by
closed-day zeros (25 December, 1 January) and the quiet days after Christmas (in the backtest the stand-in for 31 December 2025 was 0; the live dry run issued 28 December 2026 gives 4,600 and 5,200 for 30 and 31 December
2026, when earlier years sold about 12,500 and 9,800). Kevin (9 November 2026): test better stand-ins.

Six stand-ins, each served on the fold's test days exactly as in Series 87 (the models are fitted on the real training rows, N0 of Series 42; only the test rows' `forecast_sales` differs; the history is the training rows,
so nothing from the future is used; the year-ago numbers come from the full daily series and are all before the fold):
  S0  the current stand-in (reproduces Series 87)
  S1  the closed-day-aware median: the same recipe with the closed days (manager number 0) left out of the history, and 0 on the closed days themselves (25 December, 1 January)
  S2  the manager's number 364 days earlier (the same WEEKDAY a year ago) inside the holiday window 24 December to 15 January, S0 elsewhere
  S3  the manager's number 365 days earlier (the same DATE a year ago, so Christmas Day meets Christmas Day) inside the holiday window, S0 elsewhere
  S4  the hybrid: S1 everywhere and S3 inside the holiday window
  S5  S1 everywhere, and inside the holiday window the number 364 days earlier (the same weekday a year ago) unless that day was closed, in which case S1 (added after the dry run showed S3's weekday
      misalignment on weekends and S2's closed-day misses)
and the real number for reference. A year-ago number that does not exist (before the data starts on 2 January 2024) falls back to S0.
Three model variants, as Series 87 (the unmasked package, the 1 to 8 January mask, the companion without the ten history inputs), 25 seeds (200-224), 10 folds, 585 days, the deployed weekend correction; 3 x 4 pieces x 5 tasks
= 60 tasks; each task predicts every stand-in. A sanity step checks that S0 reproduces Series 87's stored forecast to the penny.

READING (set before any result was seen)
  question 1   which stand-in serves the UNMASKED package best. Measure: its average daily miss over the 585 days and over the 23 test days of the holiday window (24-31 December 2025 and 1-15 January 2026), against
               S0's (945 over all days with S0, 854 with the real number, as the floor). A stand-in is BETTER than S0 if it lowers the all-days miss by at least 5 pounds a day AND lowers the holiday-window miss;
               NO BETTER otherwise. It is judged by the all-days figure so that a stand-in that wins the window and loses elsewhere does not pass.
  question 2   with the best stand-in, do the model tricks still help? The specialist windows (the masked model used only on 1-8 January and the companion only on 26-31 December, the unmasked model everywhere else) and
               the whole-model mask, each against the unmasked package served with the SAME stand-in: STILL HELPS if the net change is at most -2.0 (25-seed noise is about 2), otherwise NO LONGER HELPS
  limits       one December and one January in the backtest (the holiday window is 23 test days); 25 seeds; the weather is the real weather; the stand-in is built from the fold's training rows (staler than the live log's);
               the same 585 days; a different stand-in changes an input of the pre-registered log, so it needs an amendment first, but no refit

Run:  python -m src.experiments.run_series88_better_standin list | sanity | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series81_closed_days_in_lags as s81
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series87_companion_without_manager as s87
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features.serving_features import seasonal_medians
from src.models.final_model import FINAL_PARAMS, TARGET_COL

OUT_DIR = CONFIG.path("experiments_dir") / "series88"
RESULTS = OUT_DIR / "series88_better_standin.jsonl"
MANAGER_COL = "forecast_sales"
FULL_PATH = CONFIG.path("features_dir") / "engineered_features.csv"
SEEDS25, PIECES = s81.SEEDS25, s81.PIECES
STANDINS = ["S0", "S1", "S2", "S3", "S4", "S5", "real"]
HOLIDAY = ((12, 24), (1, 15))
CLOSED_DAYS = ((12, 25), (1, 1))
VARIANTS = s87.VARIANTS  # sd_base, sd_j1_8, sd_companion: the feature lists of Series 87
_FULL: dict = {}


def manager_series() -> pd.Series:
    if "m" not in _FULL:
        full = pd.read_csv(FULL_PATH, parse_dates=[DATE_COL]).sort_values(DATE_COL).set_index(DATE_COL)
        _FULL["m"] = full[MANAGER_COL].astype(float)
    return _FULL["m"]


def closed_target(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    out = np.zeros(len(d), dtype=bool)
    for month, day in CLOSED_DAYS:
        out |= ((d.dt.month == month) & (d.dt.day == day)).to_numpy()
    return out


def year_ago(dates: pd.Series, days: int) -> np.ndarray:
    m = manager_series()
    return m.reindex(pd.to_datetime(dates) - pd.Timedelta(days=days)).to_numpy()


def standins(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Every stand-in for the fold's test rows, from the training rows (history) and the year-ago manager numbers."""
    dates = test[DATE_COL]
    s0 = seasonal_medians(train, dates, MANAGER_COL)
    history = train[train[MANAGER_COL] > 0]  # closed days (manager number 0) left out
    s1 = seasonal_medians(history, dates, MANAGER_COL)
    s1 = np.where(closed_target(dates), 0.0, s1)
    window = s82.in_window(dates, *HOLIDAY)
    s2 = np.where(window & ~np.isnan(year_ago(dates, 364)), year_ago(dates, 364), s0)
    s3 = np.where(window & ~np.isnan(year_ago(dates, 365)), year_ago(dates, 365), s0)
    s4 = np.where(window & ~np.isnan(year_ago(dates, 365)), year_ago(dates, 365), s1)
    ya364 = year_ago(dates, 364)
    s5 = np.where(window & ~np.isnan(ya364) & (ya364 > 0), ya364, s1)
    return {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "real": test[MANAGER_COL].to_numpy(dtype=float)}


def fit_task88(task: dict) -> dict:
    """Fit once per fold and seed on the real training rows, then predict the fold's test days with every stand-in."""
    from catboost import CatBoostRegressor

    df, folds = _STATE["df"], _STATE["folds"]
    started, features, target = time.time(), task["features"], s54.PIECES[task["piece"]][1]
    out = {name: [] for name in STANDINS}
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)].copy()
        served = standins(train, test)
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        per = {name: [] for name in STANDINS}
        for seed in task["seeds"]:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y)
            for name in STANDINS:
                frame = test.copy()
                frame[MANAGER_COL] = served[name]
                p = model.predict(frame[features])
                per[name].append(np.expm1(p) if target == "log1p" else p)
        for name in STANDINS:
            out[name].append(np.array(per[name]))
    per_standin = {name: [[round(float(x), 2) for x in row] for row in np.concatenate(out[name], axis=1)] for name in STANDINS}
    return {"id": task["id"], "part": "backtest", "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"], "per_seed": per_standin["S0"], "per_standin": per_standin,
            "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for variant, features_of in VARIANTS.items():
        for piece in PIECES:
            features = features_of(piece)
            for i in range(0, len(SEEDS25), s54.BACKTEST_CHUNK):
                chunk = SEEDS25[i:i + s54.BACKTEST_CHUNK]
                out.append({"id": f"backtest_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": variant, "seeds": chunk, "features": features})
    return out


def mine(done: dict, piece: str, variant: str, standin: str) -> np.ndarray:
    rows = [np.array(r["per_standin"][standin]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(SEEDS25), (piece, variant, standin, per_seed.shape)
    return per_seed.mean(axis=0)


def sanity() -> None:
    """S0 of one task must reproduce Series 87's stored `sd_base` forecast for the same piece and seeds, and `real` must reproduce Series 81's stored `base` fit."""
    from src.features import payday_intensity as pi
    from src.forecast_log.v1 import add_hydro_thu

    piece = "Monday-Thursday (hydro)"
    s47.init_worker47()
    _STATE["df"] = pi.add_payday_intensity_fixed(add_hydro_thu(_STATE["df"]), DATE_COL)
    got = fit_task88({"id": "sanity", "piece": piece, "variant": "sd_base", "seeds": SEEDS25[:5], "features": VARIANTS["sd_base"](piece)})["per_standin"]
    s87_rows = [np.array(r["per_seed"]) for r in read_done(s87.RESULTS).values() if (r["piece"], r["variant"]) == (piece, "sd_base") and r["seeds"] == SEEDS25[:5]]
    s81_rows = [np.array(r["per_seed"]) for r in read_done(s81.RESULTS).values() if (r["piece"], r["variant"]) == (piece, "base") and r["seeds"] == SEEDS25[:5]]
    gap0, gapr = float(np.max(np.abs(np.array(got["S0"]) - s87_rows[0]))), float(np.max(np.abs(np.array(got["real"]) - s81_rows[0])))
    print(f"sanity: S0 against Series 87's stored forecast: largest difference {gap0:.4f}; real number against Series 81's base fit: {gapr:.4f}")
    assert gap0 < 0.01 and gapr < 0.01, "the stand-in loop does not reproduce the stored fits"


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    jan8 = s82.in_window(days["date"], (1, 1), (1, 8))
    hol = s82.in_window(days["date"], *HOLIDAY)
    assert six.sum() == 6 and jan8.sum() == 8, (int(six.sum()), int(jan8.sum()))
    done = read_done(RESULTS)
    pd.set_option("display.width", 250)

    def package(variant: str, standin: str, spec: bool = False) -> np.ndarray:
        main = {pc: mine(done, pc, "sd_base", standin) for pc in PIECES}
        if spec:  # the unmasked model, the masked model only on 1-8 Jan, the companion only on 26-31 Dec
            msk, comp = {pc: mine(done, pc, "sd_j1_8", standin) for pc in PIECES}, {pc: mine(done, pc, "sd_companion", standin) for pc in PIECES}
            p = {pc: np.where(six, comp[pc], np.where(jan8, msk[pc], main[pc])) for pc in PIECES}
        else:
            p = {pc: mine(done, pc, variant, standin) for pc in PIECES}
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    rows = {}
    for s in STANDINS:
        err = np.abs(package("sd_base", s) - a)
        rows[s] = {"average miss, all 585 days": float(err.mean()), f"average miss, the {int(hol.sum())} holiday-window days": float(err[hol].mean()), "average miss, the 6 days 26-31 Dec": float(err[six].mean()),
                   "average miss, the 8 days 1-8 Jan": float(err[jan8].mean())}
    t1 = pd.DataFrame(rows).T
    print("QUESTION 1: THE UNMASKED PACKAGE SERVED WITH EACH STAND-IN (pounds; lower is better; 'real' is the floor)\n")
    print(t1.round(0).to_string())
    s0_all, s0_hol = float(t1.loc["S0", "average miss, all 585 days"]), float(t1.loc["S0", f"average miss, the {int(hol.sum())} holiday-window days"])
    print()
    better = []
    for s in ("S1", "S2", "S3", "S4", "S5"):
        all_, win = float(t1.loc[s, "average miss, all 585 days"]), float(t1.loc[s, f"average miss, the {int(hol.sum())} holiday-window days"])
        ok = (s0_all - all_ >= 5.0) and (win < s0_hol)
        better.append((all_, s, ok))
        print(f"READING SET BEFOREHAND, {s}: all days {all_ - s0_all:+.1f} against S0, holiday window {win - s0_hol:+.0f}: {'BETTER than S0' if ok else 'NO BETTER than S0'}")
    best = min(better)[1]
    print(f"\nthe lowest all-days miss: {best}")
    # question 2: do the tricks still help, under each stand-in
    rows2 = {}
    for s in ("S0", "S1", "S2", "S3", "S4", "S5"):
        base = package("sd_base", s)
        for name, pred in (("whole-model mask", package("sd_j1_8", s)), ("companion on every day", package("sd_companion", s)), ("specialist windows", package("sd_base", s, spec=True))):
            d = np.abs(pred - a) - np.abs(base - a)
            rows2[(s, name)] = {"net, 585 days": float(d.mean()), "the 6 days 26-31 Dec": float(d[six].mean()), "the 8 days 1-8 Jan": float(d[jan8].mean()), "all other days": float(d[~(six | jan8)].mean())}
    t2 = pd.DataFrame(rows2).T
    print("\nQUESTION 2: DO THE MODEL TRICKS STILL HELP? Change against the unmasked package served with the SAME stand-in (pounds a day; negative = better)\n")
    print(t2.round(2).to_string())
    for s in ("S0", "S1", "S2", "S3", "S4", "S5"):
        net = float(t2.loc[(s, "specialist windows"), "net, 585 days"])
        print(f"READING SET BEFOREHAND, specialist windows with {s}: net {net:+.2f}: {'STILL HELPS' if net <= -2.0 else 'NO LONGER HELPS'}")
    # the stand-in values on the key days
    pick = np.where(six | jan8)[0]
    _m = manager_series()
    detail = days.loc[pick, ["date", "weekday", "actual"]].copy().reset_index(drop=True)
    for fold in folds:
        pass
    for s in STANDINS:
        # recompute the stand-in values for those days from the stored forecasts is not possible; rebuild them from the fold's training rows
        vals = []
        for d in detail["date"]:
            fold = next(f for f in folds if f.test_start <= d < f.test_end)
            train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
            test = df[df[DATE_COL] == d]
            vals.append(float(standins(train, test)[s][0]))
        detail[s] = vals
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1.to_csv(OUT_DIR / "standins_unmasked.csv")
    t2.to_csv(OUT_DIR / "tricks_by_standin.csv")
    detail.round(0).to_csv(OUT_DIR / "standins_late_dec_early_jan.csv", index=False)
    print("\nTHE STAND-IN VALUES ON 26-31 DECEMBER 2025 AND 1-8 JANUARY 2026 (the manager's real number is in the 'real' column)\n")
    print(detail.round(0).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} variants x {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds; each predicts {len(STANDINS)} stand-ins)")
        df, folds = load_protocol()
        fold = next(f for f in folds if f.test_start <= pd.Timestamp("2025-12-30") < f.test_end)
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[df[DATE_COL].between("2025-12-24", "2026-01-08") & (df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        s = standins(train, test)
        out = pd.DataFrame({"date": test[DATE_COL].dt.strftime("%a %d %b %Y").to_numpy(), **{k: np.round(v, 0) for k, v in s.items()}})
        print(out.to_string(index=False))
    elif step == "sanity":
        sanity()
    elif step == "run":
        run_parallel(make_tasks(), fit_task88, RESULTS, workers, s82.init_worker82, "series88")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "sanity", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
