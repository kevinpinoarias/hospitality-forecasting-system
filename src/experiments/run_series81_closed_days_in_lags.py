"""
Series 81 - what is wrong with the recent-sales inputs in early January: the zero-sales closed days inside them, or the busy Christmas level?

Series 80 found that switching off ten recent-sales inputs (lags, rolling averages, recent same-weekday sales) for 1 to 15 January improved the whole package by 6.6 pounds a day over the 585 days (fresh seeds 7.1) and beat
all ten placebo windows in other months, with the gain on the OTHER 571 days: the January rows teach the model a wrong lesson that spreads to every month. The mask mixed two possible causes:
  zeros      closed days (25 December and 1 January, zero sales in the data) sit inside the inputs: `lag_7_sales` is 0 on 8 January because 1 January is closed, and the rolling averages hold the zeros
  level      the busy Christmas fortnight inflates the averages (the 14-day input stood about 30% above the level the early-January days sold)
This series separates them. Five variants, each refitted in all four pieces of the current package (Monday-Thursday with the hydro flag, Friday, Saturday with `intensity_fixed`, Sunday), in the training rows and
the forecast days alike, from the full daily sales series (recomputed exactly as the pipeline computes them; checked in `list`):
  base          the unmasked package (refitted on these 25 seeds, so its noise matches the others)
  closed_nan    closed days set to MISSING in the sales and forecast-error series before the lags and rolling averages are taken (a lag that lands on a closed day is missing; rolling averages use the days that
                exist, from at least half the window)
  closed_fill   closed days replaced by the same weekday a week earlier (chained: 1 January 2026 takes 25 December's filled value, which took 18 December's), so the inputs see a normal day there
  jan_mask      Series 80's mask: the ten inputs missing on every row dated 1 to 15 January
  both          closed_fill, and then the January mask
The ten inputs: `lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`, `lag_1_fe`, `lag_7_fe`, `rolling_7_fe` (the 38 final-model features) and `lag_14_sales`, `lag_21_sales`, `same_wd_mean4`, `same_wd_max4` (the Friday and
Saturday pieces' own). The manager's forecast is not touched.

Protocol: 25-seed bagged averages (seeds 200-224), the earlier standard, which Kevin restored on 8 November 2026 because 100 seeds had become too slow; 10 folds, 585 days, the deployed weekend correction; 5 variants x 4 pieces
x 5 tasks of 5 seeds = 100 tasks. Everything is judged against the unmasked package refitted on the same 25 seeds.

READING (set before any result was seen)
  measure     the NET change of the post package over all 585 days (pounds a day; negative = better); also the change over the 21 days whose 14-day window holds a closed day (26 December to 15 January) and over all
              other days
  reference   jan_mask's net change on these seeds (Series 80 gave -6.57 on 100 seeds, so something near -4 to -9 is expected)
  zeros       closed_fill or closed_nan reaches at least 60% of jan_mask's net gain: the closed days explain most of it
  level       both closed variants reach under 30% of it, and jan_mask keeps its gain: the Christmas level (or something else in those rows) explains it
  both matter in between; reported as it is
  also        whether `both` beats jan_mask (a bigger gain than either alone) is reported, not judged
  stage two   only if a closed variant reaches at least 30% of jan_mask's gain: ten placebos with two ordinary days a year made missing or filled in the same way, to show what the repair does when the days are not closed days
  limits      25 seeds (wider noise than Series 80's 100: a gain of 2 or less is within it); one January in the backtest; the same 585 days; a refit would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series81_closed_days_in_lags list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel, test_days, read_done
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu
from src.models.final_model import TARGET_COL

OUT_DIR = CONFIG.path("experiments_dir") / "series81"
RESULTS = OUT_DIR / "series81_closed_days_in_lags.jsonl"
SEEDS25 = SEEDS[:25]
PIECES = ["Monday-Thursday (hydro)", "Friday", "Saturday", "Sunday"]
INPUTS = ["lag_7_sales", "rolling_7_sales", "rolling_14_sales", "lag_1_fe", "lag_7_fe", "rolling_7_fe", "lag_14_sales", "lag_21_sales", "same_wd_mean4", "same_wd_max4"]
VARIANTS = ["base", "closed_nan", "closed_fill", "jan_mask", "both"]
PREFIX = {"closed_nan": "cn_", "closed_fill": "cf_", "jan_mask": "jm_", "both": "cb_"}
CLOSED_DAYS = ((12, 25), (1, 1))
FULL_PATH = CONFIG.path("features_dir") / "engineered_features.csv"


def closed_mask(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    out = np.zeros(len(d), dtype=bool)
    for month, day in CLOSED_DAYS:
        out |= ((d.dt.month == month) & (d.dt.day == day)).to_numpy()
    return out


def january_window(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ((d.dt.month == 1) & (d.dt.day <= 15)).to_numpy()


def chain_fill(values: np.ndarray, closed: np.ndarray) -> np.ndarray:
    """Each closed day takes the value of the same weekday a week earlier (already filled if that day was closed too)."""
    v = values.astype(float).copy()
    for i in np.where(closed)[0]:
        v[i] = v[i - 7] if i >= 7 else np.nan
    return v


def inputs_from(sales: pd.Series, fe: pd.Series, loose: bool) -> pd.DataFrame:
    """The ten inputs from a daily sales and forecast-error series, as the pipeline computes them (build_features.py: row shifts and rolling means; run_series26: same-weekday lags). `loose` lets a rolling mean
    use the days that exist (from half the window) instead of requiring the whole window, which only matters when a day is missing."""
    m7, m14 = (4, 7) if loose else (7, 14)
    lags = pd.DataFrame({k: sales.shift(k) for k in (7, 14, 21, 28)})
    return pd.DataFrame({
        "lag_7_sales": sales.shift(7), "rolling_7_sales": sales.shift(1).rolling(7, min_periods=m7).mean(), "rolling_14_sales": sales.shift(1).rolling(14, min_periods=m14).mean(),
        "lag_1_fe": fe.shift(1), "lag_7_fe": fe.shift(7), "rolling_7_fe": fe.shift(1).rolling(7, min_periods=m7).mean(),
        "lag_14_sales": lags[14], "lag_21_sales": lags[21], "same_wd_mean4": lags.mean(axis=1), "same_wd_max4": lags.max(axis=1)})


def load_series() -> tuple[pd.Series, pd.Series]:
    full = pd.read_csv(FULL_PATH, parse_dates=[DATE_COL]).sort_values(DATE_COL).set_index(DATE_COL)
    sales = full[TARGET_COL].astype(float)
    fe = (full["forecast_error"] if "forecast_error" in full.columns else sales - full["forecast_sales"]).astype(float)
    assert sales.index.is_unique and (sales.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all(), "the daily series has gaps"
    return sales, fe


def variant_inputs(variant: str, sales: pd.Series, fe: pd.Series) -> pd.DataFrame:
    closed = closed_mask(pd.Series(sales.index))
    if variant == "base":
        out = inputs_from(sales, fe, loose=False)
    elif variant == "closed_nan":
        s, e = sales.copy(), fe.copy()
        s[closed], e[closed] = np.nan, np.nan
        out = inputs_from(s, e, loose=True)
    else:  # closed_fill, jan_mask, both
        s, e = sales, fe
        if variant in ("closed_fill", "both"):
            s = pd.Series(chain_fill(sales.to_numpy(), closed), index=sales.index)
            e = pd.Series(chain_fill(fe.to_numpy(), closed), index=fe.index)
        out = inputs_from(s, e, loose=True) if variant != "jan_mask" else inputs_from(s, e, loose=False)
        if variant in ("jan_mask", "both"):
            out.loc[january_window(pd.Series(out.index)), :] = np.nan
    return out


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity_fixed(frame, DATE_COL)
    sales, fe = load_series()
    new = {}
    for variant, prefix in PREFIX.items():
        cols = variant_inputs(variant, sales, fe).reindex(pd.DatetimeIndex(out[DATE_COL]))
        for c in INPUTS:
            new[prefix + c] = cols[c].to_numpy()
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def piece_features(piece: str, variant: str) -> list[str]:
    features = s54.piece_features(piece, "none") + ([pi.COLUMN_FIXED] if piece == "Saturday" else [])
    if variant == "base":
        return features
    return [PREFIX[variant] + f if f in INPUTS else f for f in features]


def init_worker81() -> None:
    s47.init_worker47()
    clash = {p + c for p in PREFIX.values() for c in INPUTS} & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for variant in VARIANTS:
        for piece in PIECES:
            features = piece_features(piece, variant)
            for i in range(0, len(SEEDS25), s54.BACKTEST_CHUNK):
                chunk = SEEDS25[i:i + s54.BACKTEST_CHUNK]
                out.append({"id": f"backtest_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": variant, "seeds": chunk, "features": features})
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
    d = days["date"]
    jan_days = ((d >= "2026-01-02") & (d <= "2026-01-15")).to_numpy()
    affected = (((d >= "2025-12-26") & (d <= "2026-01-15")) | (d == "2025-01-15")).to_numpy()  # days whose 14-day window holds a closed day (in the test period)
    done = read_done(RESULTS)
    pd.set_option("display.width", 250)

    def package(variant: str) -> np.ndarray:
        p = {name: mine(done, name, variant) for name in PIECES}
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base = package("base")
    rows = {}
    for variant in VARIANTS[1:]:
        diff = np.abs(package(variant) - a) - np.abs(base - a)
        rows[variant] = {"net, 585 days": float(diff.mean()), f"the {int(affected.sum())} days with a closed day in their window": float(diff[affected].mean()),
                         "all other days": float(diff[~affected].mean()), "2-15 Jan 2026 (14 days)": float(diff[jan_days].mean())}
    table = pd.DataFrame(rows).T
    print("EACH VARIANT MINUS THE UNMASKED PACKAGE, refitted on the same 25 seeds (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    jan_net = float(table.loc["jan_mask", "net, 585 days"])
    print()
    share = {}
    for variant in ("closed_nan", "closed_fill", "both"):
        share[variant] = float(table.loc[variant, "net, 585 days"]) / jan_net if jan_net < 0 else float("nan")
        print(f"{variant}: net {table.loc[variant, 'net, 585 days']:+.2f} = {share[variant] * 100:.0f}% of the January mask's {jan_net:+.2f}")
    best = max(share["closed_nan"], share["closed_fill"])
    verdict = "ZEROS: the closed days explain most of it" if best >= 0.6 else "LEVEL: the closed days explain little, the January mask keeps its gain" if (best < 0.3 and jan_net < 0) else "BOTH MATTER (in between)"
    print(f"\nREADING SET BEFOREHAND: {verdict}")
    print(f"`both` against jan_mask: {table.loc['both', 'net, 585 days'] - jan_net:+.2f} (negative = a bigger gain than the mask alone)")
    print(f"stage two (placebo days) {'IS' if best >= 0.3 else 'is NOT'} triggered by the reading")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "variants.csv")
    pick = jan_days | (d == "2026-01-01").to_numpy()
    detail = days.loc[pick, ["date", "weekday", "actual"]].copy()
    for v in VARIANTS:
        detail[v] = package(v)[pick]
    detail.round(0).to_csv(OUT_DIR / "january_days.csv", index=False)
    print("\nTHE PACKAGE'S FORECAST ON 1 TO 15 JANUARY 2026:\n")
    print(detail.round(0).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} variants x {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds)")
        s47.init_worker47()
        frame = add_columns(add_hydro_thu(_STATE["df"]))
        sales, fe = load_series()
        base = variant_inputs("base", sales, fe).reindex(pd.DatetimeIndex(frame[DATE_COL]))
        for c in INPUTS:  # the recomputed base inputs equal the ones the pipeline fitted on, wherever both exist
            assert np.allclose(base[c].to_numpy(), frame[c].to_numpy(), equal_nan=True, atol=1e-6), f"the recomputed {c} differs from the pipeline's"
        print("the recomputed unmasked inputs equal the pipeline's own, on every row of the modelling frame")
        closed = closed_mask(frame[DATE_COL])
        dt = frame[DATE_COL]
        for variant, prefix in PREFIX.items():
            changed = np.zeros(len(frame), dtype=bool)
            for c in INPUTS:
                changed |= ~np.isclose(frame[prefix + c].to_numpy(), frame[c].to_numpy(), equal_nan=True)
            by_year = dt[changed].dt.year.value_counts().sort_index().to_dict()
            print(f"{variant:12s} changes the inputs on {int(changed.sum()):3d} rows ({by_year}); first {dt[changed].min():%d %b %Y}, last {dt[changed].max():%d %b %Y}")
        r = frame[frame[DATE_COL].isin(pd.to_datetime(["2026-01-05", "2026-01-08", "2026-01-12"]))]
        print("\nlag_7_sales and rolling_14_sales on three January days, by variant:")
        for variant, prefix in {"base": "", **PREFIX}.items():
            print(f"  {variant:12s}", {f"{x:%d %b}": (round(float(lag), 0), round(float(rr), 0)) for x, lag, rr in zip(r[DATE_COL], r[prefix + "lag_7_sales"], r[prefix + "rolling_14_sales"])})
        print(f"closed days in the frame: {[f'{x:%d %b %Y}' for x in dt[closed]]}")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker81, "series81")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
