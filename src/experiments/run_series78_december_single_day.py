"""
Series 78 - December with the 15th removed and a single extra pay day (the 20th or the 27th), on top of the closed-day fix; and the Boxing Day penalty four times as large.

Series 77 kept the closed-day fix (25 December and 1 January at 0, Boxing Day at -0.5) as the way forward: alone it improved the post model's net change by 1.58 pounds a day against the current intensity (fresh seeds 1.54);
the 20th as an early pay date added nothing on top of it. Kevin's next version (8 November 2026), tested here in steps so that each change can be seen on its own:

  xmas_only (reference)   the fix alone: Series 77's column, seeds 200-299 and 300-399 (reused)
  dec_base                the fix, and the 15th-of-the-month rule switched off in December only (its heat on 10 to 18 December is gone, 2024's payday having rolled back to Fri 13; every other month keeps it)
  dec_x20_tri / dec_x20_flat   dec_base plus a pay day on the 20th of December, rolled back to the previous working day (Fri 20 Dec 2024, Fri 19 Dec 2025, Fri 18 Dec 2026), share 0.34;
                          tri = the usual triangle (1 - k / 4 for 0 to 3 days either side), flat = that day only, no radius
  dec_x27_tri / dec_x27_flat   the same with the 27th of December (the date itself, not rolled back: Fri 27 Dec 2024, Sat 27 Dec 2025, Sun 27 Dec 2026)
  xmas_box4               xmas_only with the Boxing Day penalty four times as large (-2.0 instead of -0.5)
  "27" was said without saying whether the 27th or the 20th was meant, so both are run; the 20th and 27th triangles are the pay-day shapes the other rules use; the flat form is the single-day alternative Kevin asked about.

The Boxing Day penalty is a fixed placeholder (-0.5), chosen only to sit below every other value in the column (the intensity otherwise runs from 0 to about 1.1). A tree model splits on the ORDER of a column's values, so -0.5 and -2.0
should give the same trees: xmas_box4 is expected to be identical to xmas_only to the last pound.

Same design as Series 76-77: the Saturday piece with the intensity column swapped, 585 days, 10 folds, seeds 200-299 and the same columns on fresh seeds 300-399, backtest only; 6 columns x 2 seed sets x 20 tasks = 240 tasks.
Everything is judged by the NET change of the post model over all 585 days (pounds a day; negative = better), with the four December 2025 Saturdays and the other Saturdays shown next to it.

READING (set before any result was seen)
  dec_base    against xmas_only: helps if the net change is at most -0.3 and the fresh-seed net change is below zero; hurts if the net change is +0.5 or more; nothing if it is within 0.3 of zero; otherwise in between
  the four single-day columns   each against dec_base (the column it is built on): adds value if the net change is at most -0.3, the fresh-seed net change is below zero and the December-Saturday change is below zero;
              worse if the net change is +0.5 or more; otherwise nothing / in between. Only columns that add value go on to the placebo stage (eight wrong December dates, as Series 76-77); no placebos are run here.
  xmas_box4   identical to xmas_only (largest forecast difference under 1 pound) = a tree only sees the order of the values; otherwise reported as it is
  expected    dec_base within noise or slightly better (the 15th's December bump was small); the single-day columns within noise (Series 76-77 found no signal for an early date); box4 identical
  limits      one December in the backtest and one in training; 'adds value' here is only permission to run the placebos, not a result; the same 585 days throughout; v1.11 is not changed

Run:  python -m src.experiments.run_series78_december_single_day list | run [--workers 10] | report
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
from src.experiments import run_series77_christmas_closed_days as s77
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series78"
RESULTS = OUT_DIR / "series78_december_single_day.jsonl"
PIECE = "Saturday"
FRESH_SEEDS = list(range(300, 400))
SHARE = s76.EARLY_SHARE
BOX4 = 4 * s77.BOXING_DAY_PENALTY
REAL = ["dec_base", "dec_x20_tri", "dec_x20_flat", "dec_x27_tri", "dec_x27_flat", "xmas_box4"]
PARENT = {"dec_base": None, "dec_x20_tri": "dec_base", "dec_x20_flat": "dec_base", "dec_x27_tri": "dec_base", "dec_x27_flat": "dec_base", "xmas_box4": None}  # None = xmas_only


def rule_paydays(rule, years: range, skip_december: bool) -> np.ndarray:
    out = []
    for y in years:
        for m in range(1, 13):
            if skip_december and m == 12:
                continue
            out.append(pi.month_payday(pd.Period(f"{y}-{m:02d}", "M"), rule, s76.CALENDAR))
    return np.sort(np.array(out, dtype="datetime64[ns]"))


def base_intensity(dates: pd.Series, drop_15_in_december: bool) -> np.ndarray:
    """`intensity_sym`, optionally without the 15th-of-the-month rule in December."""
    d = pd.to_datetime(dates)
    years = range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2)
    total = np.zeros(len(d))
    for rule in pi.RULES:
        total += pi.SHARES[rule] * s76.heat(d, rule_paydays(rule, years, skip_december=(rule == 15 and drop_15_in_december)))
    return total


def single_day(dates: pd.Series, day: int, form: str) -> np.ndarray:
    """An extra December pay day (share 0.34): the 20th rolled back off weekends and bank holidays, or the 27th as it stands; a triangle of heat around it, or that day only."""
    d = pd.to_datetime(dates)
    years = range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2)
    paydays = np.sort(np.array([s76.early_a(y) if day == 20 else pd.Timestamp(y, 12, day) for y in years], dtype="datetime64[ns]"))
    if form == "tri":
        return SHARE * s76.heat(d, paydays)
    return SHARE * np.isin(d.dt.normalize().to_numpy().astype("datetime64[ns]"), paydays).astype(float)


def fix(dates: pd.Series, values: np.ndarray, penalty: float) -> np.ndarray:
    out = np.array(values, dtype=float)
    for month, day in s77.CLOSED_DAYS:
        out[s77.on_day(dates, month, day)] = 0.0
    out[s77.on_day(dates, *s77.BOXING_DAY)] = penalty
    return out


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity(frame, DATE_COL)
    d = out[DATE_COL]
    base = base_intensity(d, True)
    out["dec_base"] = fix(d, base, s77.BOXING_DAY_PENALTY)
    for day in (20, 27):
        for form in ("tri", "flat"):
            out[f"dec_x{day}_{form}"] = fix(d, base + single_day(d, day, form), s77.BOXING_DAY_PENALTY)
    out["xmas_box4"] = fix(d, base_intensity(d, False), BOX4)
    return out


def piece_features(column: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[PIECE][0] + [column]


def init_worker78() -> None:
    s47.init_worker47()
    clash = set(REAL) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []

    def add(variant: str, column: str, seeds: list[int]) -> None:
        features = piece_features(column)
        for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
            chunk = seeds[i:i + s54.BACKTEST_CHUNK]
            out.append({"id": f"backtest_{PIECE}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": PIECE, "variant": variant, "seeds": chunk, "features": features})

    for name in REAL:
        add(name, name, SEEDS)
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
    done, done77, done73 = read_done(RESULTS), read_done(s77.RESULTS), read_done(s73.RESULTS)
    pd.set_option("display.width", 250)

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def against(sat_pred: np.ndarray, reference: np.ndarray) -> dict:
        d = np.abs(sat_pred - a) - np.abs(reference - a)
        diff = np.abs(package(sat_pred) - a) - np.abs(package(reference) - a)
        return {"Dec 2025 Saturdays (4)": float(d[sat_dec].mean()), "other Saturdays (80)": float(d[sat_other].mean()), "post model, 31 Dec days": float(diff[dec25].mean()), "post model, all 585 days (net)": float(diff.mean())}

    net, dsat = "post model, all 585 days (net)", "Dec 2025 Saturdays (4)"
    fix_only = s76.mine(done77, "xmas_only", SEEDS)
    fix_only_fresh = s76.mine(done77, "xmas_only_fresh", FRESH_SEEDS)
    a0 = base[(PIECE, "intensity_sym")]
    preds = {n: s76.mine(done, n, SEEDS) for n in REAL}
    fresh = {n: s76.mine(done, f"{n}_fresh", FRESH_SEEDS) for n in REAL}
    refs = {None: (fix_only, fix_only_fresh)} | {n: (preds[n], fresh[n]) for n in REAL}

    def block(by_parent: bool) -> pd.DataFrame:
        rows = {}
        for n in REAL:
            parent = PARENT[n] if by_parent else None
            ref, ref_fresh = refs[parent]
            r = against(preds[n], ref)
            f = against(fresh[n], ref_fresh)
            rows[n] = r | {"net, fresh seeds": f[net], "Dec Saturdays, fresh seeds": f[dsat]}
        return pd.DataFrame(rows).T

    print("EACH COLUMN MINUS xmas_only (the closed-day fix alone; pounds a day; negative = better; seeds 200-299)\n")
    vs_fix = block(False)
    print(vs_fix.round(2).to_string())
    print("\nEACH COLUMN MINUS THE COLUMN IT IS BUILT ON (dec_base - xmas_only; the four single-day columns - dec_base; xmas_box4 - xmas_only)\n")
    vs_parent = block(True)
    print(vs_parent.round(2).to_string())
    print()
    verdicts = {}
    for n in REAL:
        r = vs_parent.loc[n]
        nt, fr, ds = float(r[net]), float(r["net, fresh seeds"]), float(r[dsat])
        if n == "xmas_box4":
            gap = float(np.max(np.abs(preds[n] - fix_only)))
            verdicts[n] = f"IDENTICAL (largest forecast difference {gap:.2f} pounds)" if gap < 1 else f"NOT IDENTICAL (largest forecast difference {gap:.1f} pounds; net {nt:+.2f})"
        elif n == "dec_base":
            verdicts[n] = "HELPS" if (nt <= -0.3 and fr < 0) else "HURTS" if nt >= 0.5 else "NOTHING" if abs(nt) < 0.3 else "IN BETWEEN"
        else:
            verdicts[n] = "ADDS VALUE (goes on to the placebos)" if (nt <= -0.3 and fr < 0 and ds < 0) else "WORSE" if nt >= 0.5 else "NOTHING" if abs(nt) < 0.3 else "IN BETWEEN"
        print(f"READING SET BEFOREHAND, {n}: {verdicts[n]}")

    pick = days["date"].isin(pd.to_datetime(["2025-12-16", "2025-12-19", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-31"])).to_numpy() | sat_dec
    detail = days.loc[pick, ["date", "weekday", "actual"]].copy()
    detail["current"] = a0[pick]
    detail["fix only"] = fix_only[pick]
    for n in REAL:
        detail[n] = preds[n][pick]
    detail = detail.sort_values("date")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vs_fix.to_csv(OUT_DIR / "vs_fix_alone.csv")
    vs_parent.to_csv(OUT_DIR / "vs_parent.csv")
    detail.round(0).to_csv(OUT_DIR / "december_days.csv", index=False)
    print("\nTHE SATURDAY PIECE'S FORECAST ON DECEMBER 2025 DAYS (only Saturdays enter the package):\n")
    print(detail.round(0).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(REAL)} real columns, each on seeds 200-299 and fresh seeds 300-399: {len(tasks)} tasks (backtest only, Saturday piece)")
        d, _ = load_protocol()
        d = d.copy()
        d[DATE_COL] = pd.to_datetime(d[DATE_COL])
        frame = add_columns(d)
        assert frame[REAL].notna().all().all(), "a column has missing values"
        s77_frame = s77.add_columns(d)
        dt = frame[DATE_COL]
        special = s77.on_day(dt, 12, 25) | s77.on_day(dt, 1, 1) | s77.on_day(dt, 12, 26)
        in_dec = (dt.dt.month == 12).to_numpy()
        # xmas_box4 = xmas_only except for the penalty value
        assert np.allclose(frame["xmas_box4"][~special], s77_frame["xmas_only"][~special]), "xmas_box4 differs from xmas_only beyond the penalty"
        assert np.allclose(frame.loc[s77.on_day(dt, 12, 26), "xmas_box4"], BOX4) and np.isclose(BOX4, -2.0)
        # dec_base = xmas_only without the 15th's December heat: only 12 to 18 December differ, and only downwards
        diff = frame["dec_base"].to_numpy() - s77_frame["xmas_only"].to_numpy()
        changed = np.abs(diff) > 1e-12
        assert changed.any() and (dt[changed].dt.month == 12).all() and dt[changed].dt.day.between(10, 18).all() and (diff[changed] < 0).all(), "dec_base differs from xmas_only somewhere it should not"
        for n in REAL[1:5]:
            extra = frame[n].to_numpy() - frame["dec_base"].to_numpy()
            ch = np.abs(extra) > 1e-12
            assert ch.any() and (dt[ch].dt.month == 12).all() and (extra[ch] > 0).all() and not special[ch].any(), f"{n} differs from dec_base somewhere it should not"
        print("checks passed: dec_base differs from xmas_only only on 10-18 December (downwards); the single-day columns differ from dec_base only in December, upwards, and never on 25/26 December or 1 January; xmas_box4 differs from xmas_only only on 26 December")
        sample = frame[frame[DATE_COL].isin(pd.to_datetime(["2025-12-15", "2025-12-18", "2025-12-19", "2025-12-20", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-31"]))]
        out = sample[[DATE_COL, pi.COLUMN] + REAL].copy()
        out["xmas_only"] = s77_frame.loc[sample.index, "xmas_only"]
        out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d %a")
        print("\nthe intensity on some December 2025 days (intensity_sym = today's; xmas_only = the fix alone):")
        print(out[[DATE_COL, pi.COLUMN, "xmas_only"] + REAL].round(2).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker78, "series78")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
