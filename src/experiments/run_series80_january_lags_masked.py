"""
Series 80 - what the package predicts for early January when every recent-sales input is switched off for 1 to 15 January, and whether the January rows are special.

Kevin's idea (8 November 2026): in early January the recent-sales inputs (last week's sales, the 7- and 14-day averages and the like) still contain the busy Christmas fortnight, so they overstate the level; switch them
off from 1 to 15 January and let the model forecast those days from the calendar, the weather and the rest, then let it use them again from the 16th. The check before this series (no fits): over the 14 open days of 2 to 15 January
the 14-day average input stood about 1,700 to 2,000 pounds (about 30%) above the level the days actually sold (2025: 8,457 against 6,484; 2026: 7,485 against 5,760), and on the 2026 Monday-Thursday days the forecast error followed
that input (correlation +0.90, 8 days) but in 2025 it did not (+0.34): plausible, not shown.

What is masked: every lag, rolling and recent-same-weekday input a piece uses: `lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`, `lag_1_fe`, `lag_7_fe`, `rolling_7_fe` (the 38 final-model features) and `lag_14_sales`, `lag_21_sales`,
`same_wd_mean4`, `same_wd_max4` (the Friday and Saturday pieces' own). They are set to missing (NaN, which CatBoost treats as its own value) on every row dated 1 to 15 January, in the TRAINING rows as well as the days being
forecast, so the model learns how to forecast those days without them (from January 2025's rows; January 2024 is not in the training rows). The manager's forecast (`forecast_sales`) is not masked: it is an input of its own,
and it was over-forecast in the window too (+693 a day in 2026).

Pieces: the four of the current package (Monday-Thursday with the hydro flag, Friday, Saturday with the closed-day column `intensity_fixed`, Sunday), each refitted with the masked inputs, 100 seeds (200-299), backtest only
(585 days, 10 folds, as Series 54). Judged against the same package unmasked (the Series 54/77 fits on the same seeds: reused), after the deployed weekend correction.

STAGE ONE, RESULT (8 November 2026, before stage two was designed): the 14 open days of 2 to 15 January 2026 -42 pounds a day (1,424 to 1,382); the other 571 days -5.71; net over 585 days -6.57. By the stage-one reading
(helps: the 14-day change at most -100 and the other days within +-1) that is NOTHING; but the other 571 days improved by more than the intensity (-3.0) or the closed-day fix (-1.6) did, which this series did not expect.
The first explanation: the January rows teach the model a wrong lesson (high recent sales, lean sales) that spreads to every month, and masking them removes it. The second: refit noise, larger than before because this is
the first test that refits all four pieces at once.

STAGE TWO (written before it ran): to tell the two apart,
  placebos    the same four pieces masked the same way for the 1st to the 15th of each other month, February to November (ten windows; training rows and forecast days alike)
  fresh seeds the January mask again on seeds 300-399, against the unmasked package refitted on 300-399 (Saturday: Series 77's `xmas_only_fresh`); 3 + 4 pieces x 20 tasks
  tasks      800 placebo tasks + 140 fresh-seed tasks = 940, plus stage one's 80
READING for stage two
  measure    the NET change of the post package over all 585 days (pounds a day; negative = better); for each placebo also the change over its own window's test days and over all the other days
  special    January's net change beats at least 9 of the 10 placebos AND the fresh-seed net change is below zero and at least half the size of the seed-200 one: the January rows are special (explanation 1)
  noise      January's net change is as good as at least 3 of the 10 placebos: refit noise (explanation 2); nothing is concluded about January
  in between reported as it is
  also       the placebo windows' own-window change is reported: a mask in a month where the recent-sales inputs are honest should hurt there; if it does not, the inputs carry little in the first two weeks of any month
  limits     one January in the backtest and two in all; the masked branch has about 15 training rows per window; the manager's forecast is left in; v1.12 is not changed by this

Run:  python -m src.experiments.run_series80_january_lags_masked list | run [--workers 10] | report
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

OUT_DIR = CONFIG.path("experiments_dir") / "series80"
RESULTS = OUT_DIR / "series80_january_lags_masked.jsonl"
FRESH_SEEDS = list(range(300, 400))
MASKED = ["lag_7_sales", "rolling_7_sales", "rolling_14_sales", "lag_1_fe", "lag_7_fe", "rolling_7_fe", "lag_14_sales", "lag_21_sales", "same_wd_mean4", "same_wd_max4"]
PIECES = ["Monday-Thursday (hydro)", "Friday", "Saturday", "Sunday"]
LAST_DAY = 15
PLACEBO_MONTHS = list(range(2, 12))
# variant -> (month masked, column prefix). "masked" is January, as stage one ran it (its prefix is kept so stage one's results stay valid)
WINDOWS = {"masked": (1, "m_"), **{f"plc_m{m:02d}": (m, f"p{m:02d}_") for m in PLACEBO_MONTHS}}
NEW_COLUMN_NAMES = [prefix + c for _, prefix in WINDOWS.values() for c in MASKED]


def in_window(dates: pd.Series, month: int) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ((d.dt.month == month) & (d.dt.day <= LAST_DAY)).to_numpy()


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity_fixed(frame, DATE_COL)
    for month, prefix in WINDOWS.values():
        w = in_window(out[DATE_COL], month)
        for col in MASKED:
            masked = out[col].astype(float).copy()
            masked[w] = np.nan
            out[prefix + col] = masked
    return out


def piece_features(piece: str, variant: str | None) -> list[str]:
    """The piece's inputs; with a variant, its masked columns in the place of the originals (None = the unmasked package)."""
    features = s54.piece_features(piece, "none") + ([pi.COLUMN_FIXED] if piece == "Saturday" else [])
    if variant is None:
        return features
    prefix = WINDOWS[variant][1]
    return [prefix + f if f in MASKED else f for f in features]


def init_worker80() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMN_NAMES) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []

    def add(piece: str, variant: str, seeds: list[int], features: list[str]) -> None:
        for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
            chunk = seeds[i:i + s54.BACKTEST_CHUNK]
            out.append({"id": f"backtest_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": variant, "seeds": chunk, "features": features})

    for variant in WINDOWS:
        for piece in PIECES:
            add(piece, variant, SEEDS, piece_features(piece, variant))
    for piece in PIECES:
        add(piece, "masked_fresh", FRESH_SEEDS, piece_features(piece, "masked"))
        if piece != "Saturday":  # Saturday's unmasked fresh-seed fits are Series 77's xmas_only_fresh
            add(piece, "base_fresh", FRESH_SEEDS, piece_features(piece, None))
    return out


def mine(done: dict, piece: str, variant: str, seeds: list[int]) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(seeds), (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    window = ((days["date"] >= "2026-01-02") & (days["date"] <= "2026-01-15")).to_numpy()
    assert window.sum() == 14, int(window.sum())
    bags = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    done, done77 = read_done(RESULTS), read_done(s77.RESULTS)
    pd.set_option("display.width", 250)

    def package(mon_thu, fri, sat, sun):
        pred = np.select([dow == 4, dow == 5, dow == 6], [fri, sat, sun], default=mon_thu)
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def masked_package(variant: str, seeds: list[int], suffix: str = "") -> np.ndarray:
        name = variant + suffix
        return package(*(mine(done, p, name, seeds) for p in ("Monday-Thursday (hydro)", "Friday", "Saturday", "Sunday")))

    ref = package(bags[("Monday-Thursday (hydro)", "none")], bags[("Friday", "none")], s76.mine(done77, "xmas_only", SEEDS), bags[("Sunday", "none")])
    ref_fresh = package(mine(done, "Monday-Thursday (hydro)", "base_fresh", FRESH_SEEDS), mine(done, "Friday", "base_fresh", FRESH_SEEDS), s76.mine(done77, "xmas_only_fresh", FRESH_SEEDS),
                        mine(done, "Sunday", "base_fresh", FRESH_SEEDS))

    def changes(new: np.ndarray, reference: np.ndarray, own: np.ndarray) -> dict:
        diff = np.abs(new - a) - np.abs(reference - a)
        return {"own-window days": int(own.sum()), "own window": float(diff[own].mean()) if own.any() else float("nan"), "all other days": float(diff[~own].mean()), "net, 585 days": float(diff.mean())}

    # stage one: January 2026's 14 open days, as before
    new = masked_package("masked", SEEDS)
    err_ref, err_new = np.abs(ref - a), np.abs(new - a)
    diff = err_new - err_ref
    print("STAGE ONE: THE CURRENT PACKAGE WITH THE RECENT-SALES INPUTS SWITCHED OFF FOR 1 TO 15 JANUARY, minus the same package unmasked (pounds a day; negative = better; seeds 200-299)\n")
    print(f"  the 14 open days of 2 to 15 January 2026: {diff[window].mean():+.1f}  (average miss {err_ref[window].mean():,.0f} -> {err_new[window].mean():,.0f}; bias {np.mean(ref[window] - a[window]):+,.0f} -> {np.mean(new[window] - a[window]):+,.0f})")
    print(f"  the other 571 days: {diff[~window].mean():+.2f}      net over all 585 days: {diff.mean():+.2f}")

    # stage two: January against the ten placebo months, and on fresh seeds
    rows = {}
    for variant, (month, _) in WINDOWS.items():
        own = (in_window(days["date"], month))
        rows[variant] = changes(masked_package(variant, SEEDS), ref, own)
    table = pd.DataFrame(rows).T
    table.index = ["January (1-15)"] + [f"month {m:02d} (placebo)" for m in PLACEBO_MONTHS]
    print("\nSTAGE TWO: THE MASK IN EACH MONTH'S 1ST TO 15TH, minus the unmasked package (pounds a day; negative = better; seeds 200-299)\n")
    print(table.round(2).to_string())
    jan_net = float(table.loc["January (1-15)", "net, 585 days"])
    plc = table.iloc[1:]
    as_good = int((plc["net, 585 days"] <= jan_net).sum())
    fresh = changes(masked_package("masked", FRESH_SEEDS, "_fresh") if False else package(*(mine(done, p, "masked_fresh", FRESH_SEEDS) for p in ("Monday-Thursday (hydro)", "Friday", "Saturday", "Sunday"))), ref_fresh, in_window(days["date"], 1))
    print(f"\nJanuary's net change {jan_net:+.2f}; placebos at least as good: {as_good} of {len(plc)} (placebo mean {plc['net, 585 days'].mean():+.2f}, best {plc['net, 585 days'].min():+.2f}, worst {plc['net, 585 days'].max():+.2f})")
    print(f"fresh seeds 300-399: January net {fresh['net, 585 days']:+.2f} (own window {fresh['own window']:+.1f}, all other days {fresh['all other days']:+.2f})")
    special = as_good <= 1 and fresh["net, 585 days"] < 0 and abs(fresh["net, 585 days"]) >= 0.5 * abs(jan_net)
    noise = as_good >= 3
    print(f"READING SET BEFOREHAND: {'JANUARY ROWS ARE SPECIAL' if special else 'REFIT NOISE (nothing is concluded about January)' if noise else 'IN BETWEEN'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "windows.csv")
    detail = days.loc[window, ["date", "weekday", "actual"]].copy()
    detail["unmasked"], detail["masked"] = ref[window], new[window]
    detail.round(0).to_csv(OUT_DIR / "january_days.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(WINDOWS)} masks x {len(PIECES)} pieces on seeds 200-299, plus the January mask and the unmasked package on fresh seeds: {len(tasks)} tasks (backtest only)")
        s47.init_worker47()  # the frame a worker fits on: the protocol's rows with the weekend pieces' own inputs added
        frame = add_columns(add_hydro_thu(_STATE["df"]))
        for variant, (month, prefix) in WINDOWS.items():
            w = in_window(frame[DATE_COL], month)
            for col in MASKED:
                assert frame.loc[w, prefix + col].isna().all() and np.array_equal(frame.loc[~w, prefix + col].to_numpy(), frame.loc[~w, col].to_numpy(), equal_nan=True), f"{variant}: {col}'s mask is wrong"
            print(f"{variant:10s} masks {int(w.sum()):3d} rows (month {month}, days 1 to {LAST_DAY}); by year: {frame.loc[w, DATE_COL].dt.year.value_counts().sort_index().to_dict()}")
        for piece in PIECES:
            f = piece_features(piece, "masked")
            print(f"{piece:26s} {len(f)} inputs, {sum(x.startswith('m_') for x in f)} masked")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker80, "series80")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
