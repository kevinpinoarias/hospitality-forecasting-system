"""
Series 77 - Series 76 again, with the closed days and Boxing Day fixed in the triangle intensity.

Series 76 moved December's pay to the 20th and found only a small, undecided nudge. Kevin's next step (8 November 2026): the intensity is wrong on the days the venue is closed. The data show zero sales on 25 December
(2024, 2025) and on 1 January (2025, 2026), yet `intensity_sym` gives them 0.26 to 0.60 of pay heat (the 25th and 28th rules roll back onto Christmas Eve and spread over Christmas Day). Boxing Day is open but slow
(2024: 4,321; 2025: 2,919), and the intensity gives it heat as well. Two changes, made on top of everything in Series 76:

  closed days   25 December and 1 January: the intensity is 0 (no pay can arrive on a day the venue is shut)
  Boxing Day    26 December: the intensity is -0.5, the opposite signal (a penalty), instead of the heat it would get. The value is a placeholder: for a tree model only its being below every other value matters,
                and it is not tuned.
  Both rules use the calendar only, so a forecast for a future day can build them. They change the column on 25 and 26 December and 1 January and nowhere else (checked in `list`).

The Saturday piece is trained on EVERY day (it is only scored on Saturdays), so these rows do reach it: Christmas Day, Boxing Day and 1 January are in its training data, and Saturday 26 December 2026 will be a Boxing
Day (and Saturday 25 December 2027 a Christmas Day), days it has never seen on a Saturday.

Columns (Saturday piece, the intensity column swapped, as Series 76; against A0 = the piece with `intensity_sym`, seeds 200-299, and Series 73's fresh seeds 300-399):
  xmas_only                 the Christmas fix alone, December otherwise as it is
  xmas_replace, xmas_add    the fix plus Series 76's two early-pay versions (the 20th rolled back; the three month-end rules replaced or the 20th added with share 0.34)
  xmas_replace_p3..p10,     the same two versions with the early date at the wrong day of December (the 3rd to the 10th), 16 placebos
  xmas_add_p3..p10
  3 real columns x 100 seeds + 16 placebos x 100 seeds + the 3 real ones on fresh seeds = 440 tasks of 5 seeds, backtest only (585 days, 10 folds, as Series 54).
  A and B (the two definitions of the early date) give the same date in 2024-2026, so one fit covers both (Series 76).

READING (set before any result was seen)
  The Christmas fix (the new part), judged by what it adds to the column it is built on: xmas_only against A0, xmas_replace against Series 76's dec_replace, xmas_add against dec_add (same seeds; pounds a day on the
  post model's net change over all 585 days, negative = better):
    helps      the net change is at most -0.3 in at least 2 of the 3 comparisons AND has the same sign on fresh seeds in those
    hurts      the net change is at least +0.5 in at least 2 of the 3
    nothing    all three net changes are within 0.3 of zero
    in between reported as it is
  The early date on top (Series 76's question, repeated): the same reading as Series 76 against A0 - at least 7 of the 8 same-mode placebos worse on BOTH the 4 December 2025 Saturdays and the net change, net at
    most -0.3, and on fresh seeds the December-Saturday change has the same sign and at least half the size and the net change the same sign; "adds nothing" if the December-Saturday change is not below zero or at
    least 3 of 8 placebos are as good on either measure; "worse" if the net change is above +0.5.
  descriptive (no rule): the Saturday piece's forecast on 25, 26 December 2025 and 1 January 2026 (days it is not scored on) shows whether it learned the closed days and Boxing Day better.
  expected    the Saturday piece is scored on Saturdays only, and Christmas Day, Boxing Day and 1 January were weekdays in 2024-2026, so the net effect is expected to be within noise (+-0.3); a clearer effect would
              be a surprise. The fix is made for the days the data cannot show: Saturday 26 December 2026 and Saturday 25 December 2027.
  limits      one December in the backtest (four Saturdays) and one in training; the fix cannot be tested on a Saturday Boxing Day or Christmas Day because none exists in the data; the -0.5 is untuned; the placebos
              share the Christmas fix, so they test the early date, not the fix; no pass/fail gate; v1.11 is not changed (any of this would need a later forecast-log version and an amendment first)

Run:  python -m src.experiments.run_series77_christmas_closed_days list | run [--workers 10] | report
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
from src.experiments import run_series73_intensity_stress_tests as s73
from src.experiments import run_series76_december_early_pay as s76
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series77"
RESULTS = OUT_DIR / "series77_christmas_closed_days.jsonl"
PIECE = "Saturday"
FRESH_SEEDS = list(range(300, 400))
CLOSED_DAYS = ((12, 25), (1, 1))  # (month, day) of the days with zero sales in the data
BOXING_DAY = (12, 26)
BOXING_DAY_PENALTY = -0.5
PLACEBO_DAYS = s76.PLACEBO_DAYS
MODES = s76.MODES
REAL = ["xmas_only"] + [f"xmas_{m}" for m in MODES]
PLACEBOS = [f"xmas_{m}_p{d}" for m in MODES for d in PLACEBO_DAYS]
NEW_COLUMNS = REAL + PLACEBOS
PARENT = {"xmas_only": None, "xmas_replace": "dec_replace", "xmas_add": "dec_add"}  # the Series 76 column each real column is built on (None = the current intensity)


def on_day(dates: pd.Series, month: int, day: int) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ((d.dt.month == month) & (d.dt.day == day)).to_numpy()


def christmas_fix(dates: pd.Series, values: np.ndarray) -> np.ndarray:
    """Closed days (25 December, 1 January) get 0 and Boxing Day gets the penalty; every other day keeps its value."""
    out = np.array(values, dtype=float)
    for month, day in CLOSED_DAYS:
        out[on_day(dates, month, day)] = 0.0
    out[on_day(dates, *BOXING_DAY)] = BOXING_DAY_PENALTY
    return out


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity(frame, DATE_COL)
    d = out[DATE_COL]
    out["xmas_only"] = christmas_fix(d, s76.intensity_with_december(d, None))
    for mode in MODES:
        out[f"xmas_{mode}"] = christmas_fix(d, s76.intensity_with_december(d, mode))
        for day in PLACEBO_DAYS:
            out[f"xmas_{mode}_p{day}"] = christmas_fix(d, s76.intensity_with_december(d, mode, day))
    return out


def piece_features(column: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[PIECE][0] + [column]


def init_worker77() -> None:
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
    dec25 = ((days["date"].dt.year == 2025) & (days["date"].dt.month == 12)).to_numpy()
    sat_dec, sat_other = sat & dec25, sat & ~dec25
    assert sat_dec.sum() == 4 and dec25.sum() == 31, (int(sat_dec.sum()), int(dec25.sum()))
    base = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    other = {p: base[(p, "none")] for p in s54.PIECES}
    done, done76, done73 = read_done(RESULTS), read_done(s76.RESULTS), read_done(s73.RESULTS)
    pd.set_option("display.width", 250)

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def against(sat_pred: np.ndarray, reference: np.ndarray) -> dict:
        d = np.abs(sat_pred - a) - np.abs(reference - a)
        diff = np.abs(package(sat_pred) - a) - np.abs(package(reference) - a)
        return {"Dec 2025 Saturdays (4)": float(d[sat_dec].mean()), "other Saturdays (80)": float(d[sat_other].mean()), "post model, 31 Dec days": float(diff[dec25].mean()), "post model, all 585 days (net)": float(diff.mean())}

    net, dsat = "post model, all 585 days (net)", "Dec 2025 Saturdays (4)"
    reference = base[(PIECE, "intensity_sym")]  # A0, seeds 200-299
    fresh_ref = s73.mine(done73, PIECE, "real_fresh", FRESH_SEEDS)
    preds = {name: s76.mine(done, name, SEEDS) for name in NEW_COLUMNS}
    table = pd.DataFrame({name: against(p, reference) for name, p in preds.items()}).T
    print("A0 = THE SATURDAY PIECE WITH THE CURRENT TRIANGLE INTENSITY. EACH ROW IS THE PIECE WITH THE COLUMN NAMED, minus A0 (pounds a day; negative = better; seeds 200-299)\n")
    print(table.round(2).to_string())

    # 1. the Christmas fix: each real column against the column it is built on
    print("\nTHE CHRISTMAS FIX ON ITS OWN: each column minus the column it is built on (xmas_only - A0; xmas_replace - dec_replace; xmas_add - dec_add)\n")
    fix_rows, fresh_rows = {}, {}
    for name, parent in PARENT.items():
        p_ref = reference if parent is None else s76.mine(done76, parent, SEEDS)
        f_ref = fresh_ref if parent is None else s76.mine(done76, f"{parent}_fresh", FRESH_SEEDS)
        fix_rows[name] = against(preds[name], p_ref)
        fresh_rows[name] = against(s76.mine(done, f"{name}_fresh", FRESH_SEEDS), f_ref)
    fix = pd.DataFrame(fix_rows).T
    fix["net, fresh seeds 300-399"] = [fresh_rows[n][net] for n in fix.index]
    fix["Dec Saturdays, fresh seeds"] = [fresh_rows[n][dsat] for n in fix.index]
    print(fix.round(2).to_string())
    nets = fix[net].to_numpy()
    helps = int(((nets <= -0.3) & (fix["net, fresh seeds 300-399"].to_numpy() < 0)).sum()) >= 2
    hurts = int((nets >= 0.5).sum()) >= 2
    nothing = bool((np.abs(nets) < 0.3).all())
    print(f"\n  READING SET BEFOREHAND (the Christmas fix): {'HELPS' if helps else 'HURTS' if hurts else 'NOTHING' if nothing else 'IN BETWEEN'}")

    # 2. the early date on top (Series 76's question)
    for name in ("xmas_replace", "xmas_add"):
        mode = name.split("_")[1]
        plc = table.loc[[f"xmas_{mode}_p{d}" for d in PLACEBO_DAYS]]
        row = table.loc[name]
        print(f"\n{name} against its eight wrong-date placebos (Series 76's question)")
        as_good = {}
        for measure in (dsat, net):
            as_good[measure] = int((plc[measure] <= row[measure]).sum())
            print(f"  {measure}: {row[measure]:+.2f}; placebos at least as good: {as_good[measure]} of {len(plc)} (p = {(1 + as_good[measure]) / (1 + len(plc)):.3f}); placebo mean {plc[measure].mean():+.2f}, best {plc[measure].min():+.2f}, worst {plc[measure].max():+.2f}")
        fresh = against(s76.mine(done, f"{name}_fresh", FRESH_SEEDS), fresh_ref)
        print(f"  on fresh seeds 300-399 (against the intensity-only piece): Dec Saturdays {fresh[dsat]:+.2f}, net {fresh[net]:+.2f}")
        d_sat, n_ = float(row[dsat]), float(row[net])
        f_sat, f_net = fresh[dsat], fresh[net]
        adds = all(v <= 1 for v in as_good.values()) and n_ <= -0.3 and f_sat * d_sat > 0 and abs(f_sat) >= 0.5 * abs(d_sat) and f_net * n_ > 0
        nothing_ = d_sat >= 0 or any(v >= 3 for v in as_good.values())
        print(f"  READING SET BEFOREHAND: {'ADDS VALUE' if adds else 'WORSE' if n_ > 0.5 else 'ADDS NOTHING' if nothing_ else 'IN BETWEEN'}")

    # 3. descriptive: the days around Christmas and 1 January, and the four Saturdays
    pick = days["date"].isin(pd.to_datetime(["2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2026-01-01"])).to_numpy() | sat_dec
    detail = days.loc[pick, ["date", "weekday", "actual"]].copy()
    detail["current"] = reference[pick]
    for name in ("dec_add", "dec_replace"):
        detail[name] = s76.mine(done76, name, SEEDS)[pick]
    for name in REAL:
        detail[name] = preds[name][pick]
    detail = detail.sort_values("date")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "christmas_columns.csv")
    fix.to_csv(OUT_DIR / "christmas_fix_alone.csv")
    detail.round(0).to_csv(OUT_DIR / "days_around_christmas.csv", index=False)
    print("\nTHE SATURDAY PIECE'S FORECAST AROUND CHRISTMAS (only Saturdays enter the package; Christmas Day, Boxing Day and 1 January are not scored by this piece):\n")
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
        assert np.allclose(frame[pi.COLUMN], s70.add_intensity_columns(d)["intensity_sym"]), "the intensity differs from Series 70's"
        s76_frame = s76.add_columns(d)
        special = on_day(frame[DATE_COL], 12, 25) | on_day(frame[DATE_COL], 1, 1) | on_day(frame[DATE_COL], 12, 26)
        for name in NEW_COLUMNS:
            reference = frame[pi.COLUMN] if name == "xmas_only" else s76_frame[f"dec_{name.split('_', 1)[1]}"]
            assert np.allclose(frame.loc[~special, name], reference[~special]), f"{name} differs from its Series 76 column on a day other than 25/26 December and 1 January"
        assert np.allclose(frame.loc[on_day(frame[DATE_COL], 12, 25) | on_day(frame[DATE_COL], 1, 1), REAL], 0.0) and np.allclose(frame.loc[on_day(frame[DATE_COL], 12, 26), REAL], BOXING_DAY_PENALTY)
        print("every column equals its Series 76 column except on 25 and 26 December and 1 January; closed days are 0 and Boxing Day is", BOXING_DAY_PENALTY)
        sample = frame[frame[DATE_COL].isin(pd.to_datetime(["2024-12-24", "2024-12-25", "2024-12-26", "2025-01-01", "2025-12-24", "2025-12-25", "2025-12-26", "2026-01-01"]))]
        print("\nthe intensity before and after the fix:")
        out = sample[[DATE_COL, pi.COLUMN] + REAL].copy()
        out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d %a")
        print(out.round(2).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker77, "series77")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
