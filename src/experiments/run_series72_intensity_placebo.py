"""
Series 72 - does the triangle payday intensity carry anything about payday? A placebo test with shifted dates and noise.

Series 70-71: the triangle intensity (`intensity_sym`; the four pay rules each add a triangle of heat around their payday, weighted by survey shares) is 3.34 pounds a day better (post model, after the
correction), nearly all of it in the Saturday piece. A look at where the gain sits (7 November 2026, no fits) found it is NOT concentrated on payday-adjacent days: the 23 Saturdays after a Friday payday
carry 36% of the Saturday gain and the other 61 Saturdays 64%, the average effect is 27 pounds a day on payday Saturdays and 18 on the rest, and the biggest single gains are on Saturdays with intensity
near zero (4 July 2026 at 0.00, 20 December 2025 at 0.00). That fits Series 63, where even a redundant column moved a seeded fit by about 3 pounds: the column may be a generic nudge to the Saturday model.

The test, as Series 41 did for the Hydro flag: build columns that look like the real one but carry no payday information, fit them in the Saturday piece exactly as the real one was fitted (same
columns, folds and seeds 200-299), and see whether the real column stands out.

  12 date-shifted placebos   the real intensity moved by a whole number of weeks, plc_p7 = the real value 7 days earlier, plc_m7 = 7 days later, and likewise 14, 21, 35, 42 and 49 days. The shape and
                             the weekend rolling are intact, the dates are wrong. (28 days is left out: it nearly lines up with the next payday.)
  4 noise placebos           an independent random number between 0 and 0.8 for every day (seeds 72-75), the same range as the real column: what ANY extra column does.

Only the Saturday piece is fitted (30 fits per placebo, 480 in all, about 40 minutes): it is where the gain is, and the other pieces are untouched in the Series 71 `sym_saturday_only` variant. The real
column is Series 70's Saturday piece, reused (same columns, same seeds).

READING (set before any result was seen)
  measure     the Saturday piece's change in MAE on the 84 Saturdays before the correction (pounds a day; negative = better), and the post model's change after the correction on all 585 days
  payday dates carry something   the real column is better than at least 15 of the 16 placebos on BOTH measures (rank 1 or 2 of 17; an empirical p of about 0.06 or less)
  generic effect                 at least 4 of the 16 placebos are as good as the real column on either measure (p of 0.25 or more): the column helps because it is an extra column, not because of payday
  in between                     reported as it is, no verdict
  expected    generic: several placebos reach a gain close to the real one
  limits      16 placebos is a rough null distribution; the placebos share the same 585 days and seeds; no pass/fail gate; nothing in production changes

Run:  python -m src.experiments.run_series72_intensity_placebo list | run [--workers 10] | report
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
from src.experiments import run_series71_amplify_intensity as s71
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series72"
RESULTS = OUT_DIR / "series72_intensity_placebo.jsonl"
PIECE = "Saturday"
SHIFTS = [7, -7, 14, -14, 21, -21, 35, -35, 42, -42, 49, -49]  # days; positive = the real value from that many days EARLIER
SHIFT_COLUMNS = {k: f"plc_{'p' if k > 0 else 'm'}{abs(k)}" for k in SHIFTS}
NOISE_COLUMNS = {72 + i: f"plc_noise{i}" for i in range(4)}
PLACEBOS = list(SHIFT_COLUMNS.values()) + list(NOISE_COLUMNS.values())


def piece_features(column: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[PIECE][0] + [column]


def add_placebo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    d = out[DATE_COL].dt.normalize()
    days = pd.date_range(d.min() - pd.Timedelta(days=60), d.max() + pd.Timedelta(days=60), freq="D")
    daily = pd.Series(s71.triangle_intensity(pd.Series(days), 4.0), index=days)
    for k, name in SHIFT_COLUMNS.items():
        out[name] = daily.reindex(pd.DatetimeIndex(d - pd.Timedelta(days=k))).to_numpy()
    for seed, name in NOISE_COLUMNS.items():
        out[name] = np.random.default_rng(seed).uniform(0.0, 0.8, size=len(out))
    return out


def init_worker72() -> None:
    s47.init_worker47()
    clash = set(PLACEBOS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_placebo_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for column in PLACEBOS:
        features = piece_features(column)
        for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
            for i in range(0, len(SEEDS), size):
                chunk = SEEDS[i:i + size]
                out.append({"id": f"{part}_{PIECE}_{column}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": PIECE, "variant": column, "seeds": chunk, "features": features})
    return out


def saturday_bags(part: str) -> dict[str, np.ndarray]:
    base = s54.load_bags(part, s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    out = {"none": base[(PIECE, "none")], "REAL intensity_sym": base[(PIECE, "intensity_sym")]}
    done = read_done(RESULTS)
    for column in PLACEBOS:
        rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == (part, PIECE, column)]
        per_seed = np.concatenate(rows, axis=0)
        assert per_seed.shape[0] == len(SEEDS), (part, column, per_seed.shape)
        out[column] = per_seed.mean(axis=0)
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    sat_idx = dow == 5
    bt = saturday_bags("backtest")
    nd_raw = s54.load_bags("backtest", s70.RESULTS, s70.VARIANT_COLUMNS, s54.RESULTS)  # the other pieces, deployed ("none")
    other = {p: nd_raw[(p, "none")] for p in s54.PIECES}
    frame = add_placebo_columns(s70.add_intensity_columns(df))
    corr_sat = {}
    for name in ["REAL intensity_sym"] + PLACEBOS:
        col = "intensity_sym" if name.startswith("REAL") else name
        corr_sat[name] = float(np.corrcoef(frame.loc[frame[DATE_COL].dt.dayofweek == 5, col], frame.loc[frame[DATE_COL].dt.dayofweek == 5, "intensity_sym"])[0, 1])

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base_pkg = package(bt["none"])
    rows = []
    for name in ["REAL intensity_sym"] + PLACEBOS:
        sat_change = float((np.abs(bt[name] - a) - np.abs(bt["none"] - a))[sat_idx].mean())
        diff_pkg = np.abs(package(bt[name]) - a) - np.abs(base_pkg - a)
        rows.append({"column": name, "corr with real (Saturdays)": corr_sat[name], "Saturday piece, before correction": sat_change,
                     "post model, after correction": float(diff_pkg.mean()), "Saturdays improved": int((((np.abs(bt[name] - a) - np.abs(bt["none"] - a))[sat_idx]) < 0).sum())})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("SATURDAY PIECE WITH EACH COLUMN, against the deployed Saturday piece (negative = better; 84 Saturdays, 585 days for the package)\n")
    print(table.sort_values("Saturday piece, before correction").round(2).to_string(index=False))

    real = table.iloc[0]
    plc = table.iloc[1:]
    n = len(plc)
    for measure in ("Saturday piece, before correction", "post model, after correction"):
        beaten = int((plc[measure] >= real[measure]).sum())      # placebos that are NOT better than the real column
        as_good = int((plc[measure] <= real[measure]).sum())     # placebos at least as good as the real column
        print(f"\n{measure}: the real column ({real[measure]:+.2f}) is better than {beaten} of {n} placebos; {as_good} of {n} are at least as good; empirical p = {(1 + as_good) / (1 + n):.2f}")
        for label, group in (("date-shifted", plc[plc['column'].str.startswith('plc_p') | plc['column'].str.startswith('plc_m')]), ("noise", plc[plc['column'].str.startswith('plc_noise')])):
            print(f"   {label:13s} placebos: mean {group[measure].mean():+.2f}, best {group[measure].min():+.2f}, worst {group[measure].max():+.2f}")
    as_good_both = {m: int((plc[m] <= real[m]).sum()) for m in ("Saturday piece, before correction", "post model, after correction")}
    carries = all(as_good_both[m] <= 1 for m in as_good_both)
    generic = any(as_good_both[m] >= 4 for m in as_good_both)
    print("\nREADING SET BEFOREHAND:", "PAYDAY DATES CARRY SOMETHING" if carries else ("GENERIC EFFECT" if generic else "IN BETWEEN"),
          f"(placebos at least as good as the real column: {as_good_both})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "placebos.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PLACEBOS)} placebo columns ({len(SHIFT_COLUMNS)} date-shifted, {len(NOISE_COLUMNS)} noise), Saturday piece only, {len(tasks)} tasks")
        print(f"  Saturday piece: {len(s54.FINAL_FEATURES) + len(s54.PIECES[PIECE][0])} columns, each placebo +1")
        d, _ = load_protocol()
        frame = add_placebo_columns(s70.add_intensity_columns(d))
        assert frame[PLACEBOS].notna().all().all(), "a placebo column has missing values"
        sat = frame[DATE_COL].dt.dayofweek == 5
        real = frame.loc[sat, "intensity_sym"]
        print(f"{'column':12s} {'mean':>6s} {'max':>6s} {'corr with real (Saturdays)':>28s}")
        for name in ["intensity_sym"] + PLACEBOS:
            x = frame.loc[sat, name]
            print(f"{name:12s} {x.mean():6.3f} {x.max():6.2f} {float(np.corrcoef(x, real)[0, 1]):28.2f}")
        shifted = frame.set_index(DATE_COL)
        k = 7
        probe = pd.Timestamp("2025-09-26")
        assert abs(shifted.loc[probe, "plc_p7"] - shifted.loc[probe - pd.Timedelta(days=7), "intensity_sym"]) < 1e-12, "plc_p7 is not the real value 7 days earlier"
        print(f"checked: plc_p7 on {probe.date()} = real intensity on {(probe - pd.Timedelta(days=k)).date()}")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker72, "series72")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
