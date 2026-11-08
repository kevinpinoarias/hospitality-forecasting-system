"""
Series 76 - does an early December payday help the triangle intensity in the Saturday piece?

The triangle intensity (`intensity_sym`, forecast-log version v1.11's Saturday column) pays three of its four rules at the end of the month: the last working day, the 25th and the 28th. In December that puts the pay
peak on Christmas Eve and the last days of the month, while the sales data show the strongest stretch in the third week (19 to 23 December, 2024 and 2025) and employers often pay early before Christmas (CIPP: typically
19 to 23 December). Kevin's December rule (8 November 2026): December payday is the end of the third week of December. Two ways to write it, tested as defined here:

  Definitions of the early date    A = the 20th, moved back to the previous working day if it falls on a weekend or bank holiday;
                                   B = the Friday of the Monday-to-Sunday week that contains the 20th.
                                   They give the SAME date in 2024, 2025 and 2026 (Fri 20, Fri 19, Fri 18 December), so every column below is identical under A and B in these data: one fit covers both (checked in `list`).
                                   They differ only in midweek Decembers (2021-23, 2027-29), which no December we have can separate.
  dec_replace   in December the last working day, 25th and 28th rules are all paid on the early date (their combined share 0.94 sits on one day); the 15th rule is unchanged
  dec_add       the three month-end rules stay, and the early date is added as a fourth December payday with share 0.34 (the share of the last-working-day rule; an assumption, not a measurement)
  Away from December the columns equal `intensity_sym`, except for the triangles' reach over the month edges: under `replace` 1 to 3 January lose the heat of the 31 December payday, and a placebo on the 3rd of December
                 reaches back to 30 November (checked in `list`: no difference before 29 November or after 3 January).

The test mirrors Series 75: Saturday piece only, the intensity column swapped for the December column, judged against A0 = the same piece with `intensity_sym` (Series 70's fits on seeds 200-299, Series 73's on 300-399:
reused). To show what changing December's pay dates does when the new date carries no pay information, each mode also gets eight PLACEBO versions with the early date at the wrong day of the month (the 3rd to the 10th,
moved back off weekends and bank holidays). 18 columns x 100 seeds + the two real ones again on fresh seeds 300-399 = 400 tasks of 5 seeds, backtest only (585 days, 10 folds, as Series 54).

What the data can show: only one December (2025) is in the backtest, with December 2024 the only December in the training data. Four Saturdays of December 2025 are scored; this is a sanity check, not a verdict. The
real test is December 2026, and a rule must be fixed before it. The deployed payday columns (all pieces) are NOT changed here.

READING (set before any result was seen)
  measures    for each real column, change against A0 (pounds a day; negative = better): the Saturday piece on the 4 December 2025 Saturdays; the Saturday piece on the other 80 Saturdays (a December rule should leave
              them alone: this shows the noise and any damage); the post model (the package after the correction) on all 585 days (the NET effect, the main measure) and on the 31 December 2025 days
  adds value  at least 7 of the 8 same-mode placebos are worse on BOTH the December-Saturday measure and the net measure, the net change is at most -0.3, AND on fresh seeds 300-399 the December-Saturday change has
              the same sign and at least half the size and the net change has the same sign
  adds nothing  the December-Saturday change is not below zero, or at least 3 of the 8 placebos are as good on either measure
  harms       the net change is above +0.5 (reported as WORSE; the December rule would be damaging the other months by changing what the model learns from December 2024)
  in between  reported as it is
  expected    dec_replace: little or worse (it zeroes the heat on 24 to 31 December, while those days sold as much as the third week); dec_add: closer to neutral. Neither is expected to clear the placebos on four Saturdays.
  limits      one December in the backtest (four Saturdays) and one in training; the second weekend of December sold strongly in both years, so a placebo near it can look good for reasons that are not pay; A and B cannot be
              told apart; no pass/fail gate; v1.11 is not changed by this (a December change would need a later forecast-log version and an amendment to the pre-registration first)

Run:  python -m src.experiments.run_series76_december_early_pay list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series70_payday_intensity as s70
from src.experiments import run_series73_intensity_stress_tests as s73
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series76"
RESULTS = OUT_DIR / "series76_december_early_pay.jsonl"
PIECE = "Saturday"
FRESH_SEEDS = list(range(300, 400))
CALENDAR = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
MONTH_END_RULES = ("last_working_day", 25, 28)
EARLY_DAY = 20
EARLY_SHARE = 0.34
PLACEBO_DAYS = [3, 4, 5, 6, 7, 8, 9, 10]
MODES = ("replace", "add")
REAL = [f"dec_{m}" for m in MODES]
PLACEBOS = [f"dec_{m}_p{d}" for m in MODES for d in PLACEBO_DAYS]
NEW_COLUMNS = REAL + PLACEBOS


def early_a(year: int, day: int = EARLY_DAY) -> pd.Timestamp:
    """Definition A: the `day`th of December, moved back to the previous working day if it falls on a weekend or bank holiday."""
    return pi.month_payday(pd.Period(f"{year}-12", "M"), day, CALENDAR)


def early_b(year: int) -> pd.Timestamp:
    """Definition B: the Friday of the Monday-to-Sunday week that contains the 20th of December."""
    d = pd.Timestamp(year, 12, EARLY_DAY)
    return d + pd.Timedelta(days=4 - d.weekday())


def payday_dates(rule, years: range, early_day: int | None, replace: bool) -> np.ndarray:
    out = []
    for y in years:
        for m in range(1, 13):
            if m == 12 and replace and early_day is not None and rule in MONTH_END_RULES:
                out.append(early_a(y, early_day))
            else:
                out.append(pi.month_payday(pd.Period(f"{y}-{m:02d}", "M"), rule, CALENDAR))
    return np.sort(np.array(out, dtype="datetime64[ns]"))


def heat(dates: pd.Series, paydays: np.ndarray) -> np.ndarray:
    """The triangle around the nearest payday: 1 - k / 4 for k = 0 to 3 days either side, 0 from 4 days out."""
    v = pd.to_datetime(dates).to_numpy().astype("datetime64[ns]")
    latest = paydays[np.clip(np.searchsorted(paydays, v, side="right") - 1, 0, len(paydays) - 1)]
    upcoming = paydays[np.clip(np.searchsorted(paydays, v, side="left"), 0, len(paydays) - 1)]
    one = np.timedelta64(1, "D")
    k = np.minimum(np.abs((v - latest) // one), np.abs((upcoming - v) // one)).astype(int)
    return np.clip(1.0 - k / pi.WIDTH, 0.0, None)


def intensity_with_december(dates: pd.Series, mode: str | None, early_day: int = EARLY_DAY) -> np.ndarray:
    """`intensity_sym` with December changed (mode None = unchanged): 'replace' pays the three month-end rules on the early date, 'add' keeps them and adds the early date with share EARLY_SHARE."""
    d = pd.to_datetime(dates)
    years = range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2)
    total = np.zeros(len(d))
    for rule in pi.RULES:
        total += pi.SHARES[rule] * heat(d, payday_dates(rule, years, early_day, mode == "replace"))
    if mode == "add":
        early = np.sort(np.array([early_a(y, early_day) for y in years], dtype="datetime64[ns]"))
        total += EARLY_SHARE * heat(d, early)
    return total


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity(frame, DATE_COL)
    for mode in MODES:
        out[f"dec_{mode}"] = intensity_with_december(out[DATE_COL], mode)
        for day in PLACEBO_DAYS:
            out[f"dec_{mode}_p{day}"] = intensity_with_december(out[DATE_COL], mode, day)
    return out


def piece_features(column: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[PIECE][0] + [column]


def init_worker76() -> None:
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


def mine(done: dict, variant: str, seeds: list[int]) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", PIECE, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(seeds), (variant, per_seed.shape)
    return per_seed.mean(axis=0)


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
    done = read_done(RESULTS)
    done73 = read_done(s73.RESULTS)
    pd.set_option("display.width", 250)

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def against(sat_pred: np.ndarray, reference: np.ndarray) -> dict:
        d = np.abs(sat_pred - a) - np.abs(reference - a)
        diff = np.abs(package(sat_pred) - a) - np.abs(package(reference) - a)
        return {"Dec 2025 Saturdays (4)": float(d[sat_dec].mean()), "other Saturdays (80)": float(d[sat_other].mean()), "post model, 31 Dec days": float(diff[dec25].mean()), "post model, all 585 days (net)": float(diff.mean())}

    reference = base[(PIECE, "intensity_sym")]  # A0, seeds 200-299
    rows = {name: against(mine(done, name, SEEDS), reference) for name in NEW_COLUMNS}
    table = pd.DataFrame(rows).T
    print("A0 = THE SATURDAY PIECE WITH THE CURRENT TRIANGLE INTENSITY. EACH ROW IS THE PIECE WITH THE DECEMBER-CHANGED COLUMN, minus A0 (pounds a day; negative = better; seeds 200-299)\n")
    print(table.round(2).to_string())
    fresh_ref = s73.mine(done73, PIECE, "real_fresh", FRESH_SEEDS)
    verdicts = {}
    for name in REAL:
        mode = name.split("_")[1]
        plc = table.loc[[f"dec_{mode}_p{d}" for d in PLACEBO_DAYS]]
        row = table.loc[name]
        print(f"\n{name}")
        as_good = {}
        for measure in ("Dec 2025 Saturdays (4)", "post model, all 585 days (net)"):
            as_good[measure] = int((plc[measure] <= row[measure]).sum())
            print(f"  {measure}: {row[measure]:+.2f}; placebos at least as good: {as_good[measure]} of {len(plc)} (p = {(1 + as_good[measure]) / (1 + len(plc)):.3f}); placebo mean {plc[measure].mean():+.2f}, best {plc[measure].min():+.2f}, worst {plc[measure].max():+.2f}")
        fresh = against(mine(done, f"{name}_fresh", FRESH_SEEDS), fresh_ref)
        print(f"  on fresh seeds 300-399 (against the intensity-only piece on the same seeds): Dec Saturdays {fresh['Dec 2025 Saturdays (4)']:+.2f}, net {fresh['post model, all 585 days (net)']:+.2f}")
        dec_sat, net = float(row["Dec 2025 Saturdays (4)"]), float(row["post model, all 585 days (net)"])
        f_sat, f_net = fresh["Dec 2025 Saturdays (4)"], fresh["post model, all 585 days (net)"]
        adds = all(v <= 1 for v in as_good.values()) and net <= -0.3 and f_sat * dec_sat > 0 and abs(f_sat) >= 0.5 * abs(dec_sat) and f_net * net > 0
        nothing = dec_sat >= 0 or any(v >= 3 for v in as_good.values())
        verdicts[name] = "ADDS VALUE" if adds else ("WORSE" if net > 0.5 else ("ADDS NOTHING" if nothing else "IN BETWEEN"))
        print(f"  READING SET BEFOREHAND: {verdicts[name]}")
    # the December 2025 days, day by day, for the real columns and the current piece (Saturday piece applied to every day: only the Saturdays enter the package)
    detail = days.loc[dec25, ["date", "weekday", "actual"]].copy()
    detail["current"] = reference[dec25]
    for name in REAL:
        detail[name] = mine(done, name, SEEDS)[dec25]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "december_columns.csv")
    detail.round(0).to_csv(OUT_DIR / "dec2025_saturday_piece.csv", index=False)
    print("\nDECEMBER 2025, Saturday piece's forecast on every day (only Saturdays enter the package):\n")
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
        assert np.allclose(intensity_with_december(frame[DATE_COL], None), frame[pi.COLUMN]), "the unchanged December version differs from intensity_sym"
        month, day = frame[DATE_COL].dt.month, frame[DATE_COL].dt.day
        near = (month == 12) | ((month == 1) & (day <= 3)) | ((month == 11) & (day >= 29))
        for name in NEW_COLUMNS:
            assert np.allclose(frame.loc[~near, name], frame.loc[~near, pi.COLUMN]), f"{name} differs from intensity_sym away from December"
        jan = frame[(month == 1) & (day <= 3)]
        print(f"early-January days (1 to 3 Jan) that the replace columns change: {len(jan)} days in the data, largest change {float((jan['dec_replace'] - jan[pi.COLUMN]).abs().max()):.2f}")
        for y in (2024, 2025, 2026):
            assert early_a(y) == early_b(y), f"definitions A and B differ in {y}"
        print("away from December (29 Nov to 3 Jan) every column equals intensity_sym; A and B give the same early date in 2024, 2025 and 2026:", {y: f"{early_a(y):%a %d %b}" for y in (2024, 2025, 2026)})
        dec = frame[(frame[DATE_COL].dt.year == 2025) & (frame[DATE_COL].dt.month == 12) & (frame[DATE_COL].dt.dayofweek == 5)]
        print("\nDecember 2025 Saturdays, the real columns against the current intensity:")
        print(dec[[DATE_COL, pi.COLUMN] + REAL].round(2).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker76, "series76")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
