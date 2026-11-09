"""
Series 79 - a penalty on the payday intensity for the first two weeks of January (2 to 15 January), on top of the closed-day fix.

Kevin's reading (8 November 2026): after the end-of-year pay the month is lean, so the early-January days should carry a penalty in the intensity rather than a little pay heat (2 January 0.17, 3 January 0.08 from the 31 December
payday's triangle; from 12 January the 15th rule's triangle). What the data show for 2026 (Saturday piece and weekday pieces of the post package, backtest): on the 14 open days of 2 to 15 January the package is over-forecast by 504
pounds a day on average (the manager by 693), mostly the lean weekdays of 5 to 9 January (over by 1,000 to 2,700); but the intensity lives only in the Saturday pieces, so it can reach only the two Saturdays in that window,
3 January (over-forecast by 4,383) and 10 January (UNDER-forecast by 1,719): a penalty should help one and hurt the other.

Columns (Saturday piece, the intensity column swapped; each is `xmas_only`, the closed-day fix alone = forecast-log version v1.12's column, with some days overwritten):
  jan_pen_15    2 to 15 January set to a penalty (-0.25: below every heat value, above Boxing Day's -0.5 so the two stay in separate bins; untuned, and for a tree only the order matters)
  jan_pen_8     2 to 8 January only (the first week)
  jan_zero_15   2 to 15 January set to 0: the heat removed, no penalty (to see removal and penalty apart)
  plc_m02 ... plc_m11   ten PLACEBOS: the same penalty (-0.25) on the 2nd to the 15th of February, March, ... November, to show what a 14-day penalty window does when it sits where no January regime exists
  1 January stays 0 (closed). 3 real columns + 10 placebos on seeds 200-299, and the 3 real ones on fresh seeds 300-399 = 320 tasks of 5 seeds, backtest only (585 days, 10 folds, as Series 54).

Judged against xmas_only (Series 77's fits on seeds 200-299 and 300-399: reused), by the NET change of the post model over all 585 days (pounds a day; negative = better), with the two early-January Saturdays of 2026 (3 and 10 January)
and the other 82 Saturdays shown next to it. The backtest holds ONE January window (2026: Jan 2025's first half is before the backtest starts on 15 January 2025, so it is training data only).

READING (set before any result was seen)
  adds value  net change at most -0.3, at most 1 of the 10 placebos as good on the net change, the fresh-seed net change below zero, and the change on the two early-January Saturdays below zero
  nothing     the net change is within 0.3 of zero, or at least 3 of the 10 placebos are as good on the net change
  worse       the net change is +0.5 or more
  in between  reported as it is
  expected    little: the two Saturdays pull in opposite directions (3 January over-forecast, 10 January under-forecast), the weekday over-forecasts of 5 to 9 January are out of the intensity's reach, and any refit moves the
              net change by a pound or so in either direction (Series 76-78); a clear gain would be a surprise
  limits      one January window and two Saturdays in it; the intensity cannot change a weekday forecast; the same 585 days; the -0.25 is untuned; v1.12 is not changed by this (a January version would be later and need an amendment first)

Run:  python -m src.experiments.run_series79_january_penalty list | run [--workers 10] | report
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
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series79"
RESULTS = OUT_DIR / "series79_january_penalty.jsonl"
PIECE = "Saturday"
FRESH_SEEDS = list(range(300, 400))
PENALTY = -0.25
REAL = ["jan_pen_15", "jan_pen_8", "jan_zero_15"]
PLACEBO_MONTHS = list(range(2, 12))
PLACEBOS = [f"plc_m{m:02d}" for m in PLACEBO_MONTHS]
NEW_COLUMNS = REAL + PLACEBOS


def in_window(dates: pd.Series, month: int, first: int, last: int) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ((d.dt.month == month) & (d.dt.day >= first) & (d.dt.day <= last)).to_numpy()


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity(frame, DATE_COL)
    d = out[DATE_COL]
    base = s77.christmas_fix(d, s76.intensity_with_december(d, None))  # xmas_only: the closed-day fix alone

    def with_window(month: int, last: int, value: float) -> np.ndarray:
        v = base.copy()
        v[in_window(d, month, 2, last)] = value
        return v

    out["jan_pen_15"] = with_window(1, 15, PENALTY)
    out["jan_pen_8"] = with_window(1, 8, PENALTY)
    out["jan_zero_15"] = with_window(1, 15, 0.0)
    for m in PLACEBO_MONTHS:
        out[f"plc_m{m:02d}"] = with_window(m, 15, PENALTY)
    return out


def piece_features(column: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[PIECE][0] + [column]


def init_worker79() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []

    def add(variant: str, column: str, seeds: list[int]) -> None:
        features = piece_features(column)
        for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
            chunk = seeds[i:i + s54.BACKTEST_CHUNK]
            out.append({"id": f"backtest_{PIECE}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": PIECE, "variant": variant, "seeds": chunk, "features": features})

    for name in NEW_COLUMNS:
        add(name, name, SEEDS)
    for name in REAL:
        add(f"{name}_fresh", name, FRESH_SEEDS)
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    sat = dow == 5
    jan = ((days["date"] >= "2026-01-02") & (days["date"] <= "2026-01-15")).to_numpy()
    sat_jan, sat_other = sat & jan, sat & ~jan
    assert sat_jan.sum() == 2 and jan.sum() == 14, (int(sat_jan.sum()), int(jan.sum()))
    base = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    other = {p: base[(p, "none")] for p in s54.PIECES}
    done, done77 = read_done(RESULTS), read_done(s77.RESULTS)
    pd.set_option("display.width", 250)

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def against(sat_pred: np.ndarray, reference: np.ndarray) -> dict:
        d = np.abs(sat_pred - a) - np.abs(reference - a)
        diff = np.abs(package(sat_pred) - a) - np.abs(package(reference) - a)
        return {"2 Saturdays, 3 and 10 Jan (2)": float(d[sat_jan].mean()), "other Saturdays (82)": float(d[sat_other].mean()), "post model, 2-15 Jan days (14)": float(diff[jan].mean()), "post model, all 585 days (net)": float(diff.mean())}

    net, jsat = "post model, all 585 days (net)", "2 Saturdays, 3 and 10 Jan (2)"
    fix_only = s76.mine(done77, "xmas_only", SEEDS)
    fix_only_fresh = s76.mine(done77, "xmas_only_fresh", FRESH_SEEDS)
    preds = {n: s76.mine(done, n, SEEDS) for n in NEW_COLUMNS}
    table = pd.DataFrame({n: against(p, fix_only) for n, p in preds.items()}).T
    print("EACH COLUMN MINUS xmas_only (the closed-day fix alone, forecast-log version v1.12's column; pounds a day; negative = better; seeds 200-299)\n")
    print(table.round(2).to_string())
    for name in REAL:
        plc = table.loc[PLACEBOS]
        row = table.loc[name]
        as_good = int((plc[net] <= row[net]).sum())
        fresh = against(s76.mine(done, f"{name}_fresh", FRESH_SEEDS), fix_only_fresh)
        n_, js = float(row[net]), float(row[jsat])
        adds = n_ <= -0.3 and as_good <= 1 and fresh[net] < 0 and js < 0
        verdict = "ADDS VALUE" if adds else "WORSE" if n_ >= 0.5 else "NOTHING" if (abs(n_) < 0.3 or as_good >= 3) else "IN BETWEEN"
        print(f"\n{name}: net {n_:+.2f}; placebos at least as good on the net change: {as_good} of {len(plc)} (placebo mean {plc[net].mean():+.2f}, best {plc[net].min():+.2f}, worst {plc[net].max():+.2f}); "
              f"the two January Saturdays {js:+.1f} (placebo mean {plc[jsat].mean():+.1f}); fresh seeds: net {fresh[net]:+.2f}, January Saturdays {fresh[jsat]:+.1f}")
        print(f"READING SET BEFOREHAND, {name}: {verdict}")
    pick = jan | (days["date"] == "2026-01-01").to_numpy()
    detail = days.loc[pick, ["date", "weekday", "actual"]].copy()
    detail["fix only"] = fix_only[pick]
    for n in REAL:
        detail[n] = preds[n][pick]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "january_columns.csv")
    detail.round(0).to_csv(OUT_DIR / "january_days.csv", index=False)
    print("\nTHE SATURDAY PIECE'S FORECAST ON 1 TO 15 JANUARY 2026 (only the two Saturdays enter the package):\n")
    print(detail.round(0).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(REAL)} real columns + {len(PLACEBOS)} placebo columns on seeds 200-299 and the {len(REAL)} real ones on fresh seeds 300-399: {len(tasks)} tasks (backtest only, Saturday piece)")
        d, _ = load_protocol()
        d = d.copy()
        d[DATE_COL] = pd.to_datetime(d[DATE_COL])
        frame = add_columns(d)
        assert frame[NEW_COLUMNS].notna().all().all(), "a column has missing values"
        base = s77.christmas_fix(d[DATE_COL], s76.intensity_with_december(d[DATE_COL], None))
        dt = frame[DATE_COL]
        for name, (month, last, value) in {"jan_pen_15": (1, 15, PENALTY), "jan_pen_8": (1, 8, PENALTY), "jan_zero_15": (1, 15, 0.0), **{f"plc_m{m:02d}": (m, 15, PENALTY) for m in PLACEBO_MONTHS}}.items():
            w = in_window(dt, month, 2, last)
            assert np.allclose(frame.loc[~w, name], base[~w]) and np.allclose(frame.loc[w, name], value), f"{name} is not xmas_only outside its window and the value inside it"
        assert np.allclose(frame["jan_pen_15"][in_window(dt, 1, 1, 1)], 0.0), "1 January must stay 0 (closed)"
        print("checks passed: every column equals xmas_only (the closed-day fix) outside its window and the stated value inside it; 1 January stays 0")
        sample = frame[(dt >= "2026-01-01") & (dt <= "2026-01-17")][[DATE_COL, "jan_pen_15", "jan_pen_8", "jan_zero_15"]].copy()
        sample.insert(1, "intensity_sym", pi.payday_intensity(frame.loc[sample.index, DATE_COL]))
        sample[DATE_COL] = sample[DATE_COL].dt.strftime("%Y-%m-%d %a")
        print(sample.round(2).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker79, "series79")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
