"""
Series 84 - can the six late-December days (26 to 31 December) be added to the 1 to 8 January mask without losing the gain on the other days?

Series 83 (all 100 seeds and fresh seeds) confirmed two windows for switching the ten recent-sales inputs off: 1 to 8 January (-8.35 pounds a day over the 585 days) and 26 December to 8 January (-8.57). They differ
in WHERE the gain is. The six extra late-December days gain about 460 pounds a day each from the mask (the 14-day window gains 383 a day on its own days against 324 on the eight January days), but masking those rows in
training makes every other day slightly worse (+0.6 to +1.7 outside the window, against -4 to -6 for 1 to 8 January alone): in late December the recent-sales inputs are mostly accurate (the venue is busy and recent sales are
high), so masking the training rows throws good examples away. Kevin's question (9 November 2026): can the six days be included and the benefit kept? Option 1, tested here: on the late-December rows switch off only the
inputs that mislead there, keep the others on, and keep the full ten-input mask on 1 to 8 January.

Five groups of inputs (the fifth, g_fe_wd, was added after the first four had run, on Kevin's question whether the two groups that kept the outside gain add up; a group is masked on 26 to 31 December, in the training rows and the forecast days alike; all ten are masked on 1 to 8 January in every variant):
  g_levels   `rolling_7_sales`, `rolling_14_sales` (the recent AVERAGE LEVEL)
  g_lags     `lag_7_sales`, `lag_14_sales`, `lag_21_sales` (single SAME-WEEKDAY LAGS)
  g_fe       `lag_1_fe`, `lag_7_fe`, `rolling_7_fe` (the manager's recent forecast errors)
  g_wd       `same_wd_mean4`, `same_wd_max4` (the SUMMARIES of the last four same weekdays)
  g_fe_wd    the forecast-error inputs and the same-weekday summaries together (5 inputs)
A piece only has the inputs it uses (the Friday and Saturday pieces have the lag_14/lag_21/same-weekday ones; Monday-Thursday and Sunday do not), so a group touches only the pieces that carry it.
References, from Series 81/82 on the same 25 seeds (200-224): `base` (unmasked), `j1_8` (the 1 to 8 January mask) and `d26_j8` (all ten inputs masked on 26 December to 8 January).
Protocol: 25-seed bagged averages, 10 folds, 585 days, the deployed weekend correction; 4 variants x 4 pieces x 5 tasks of 5 seeds = 80 tasks.

READING (set before any result was seen)
  measure     the NET change of the post package over the 585 days against `base` (pounds a day; negative = better), the change over the 14 days from 26 December to 8 January, and over all other days
  captures    a group captures BOTH when its net change is at least 2.0 better than j1_8's (25-seed noise is about 2) AND its change over the days outside 26 December-8 January is no more than 1.0 above j1_8's
              (the gain outside must be kept)
  no gain     the net change is within 2.0 of j1_8's: nothing was gained over the shorter window
  costs       the net change is worse than j1_8's by more than 2.0
  expected    little: the late-December gain and the outside loss both come from the same masked rows, and a subset keeps less of the first and of the second; the lag groups are the likeliest to help since the
              closed Christmas Day is inside a 7-day lag for 1 to 2 January, not for 26 to 31 December
  limits      25 seeds; one December and one January in the backtest; the same 585 days; a group is one of four tries on top of six windows, so a winner is a candidate to repeat on 100 and fresh seeds, not a result;
              a refit would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series84_late_december_subsets list | run [--workers 10] | report
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

OUT_DIR = CONFIG.path("experiments_dir") / "series84"
RESULTS = OUT_DIR / "series84_late_december_subsets.jsonl"
SEEDS25, PIECES, INPUTS = s81.SEEDS25, s81.PIECES, s81.INPUTS
JAN = ((1, 1), (1, 8))
DEC = ((12, 26), (12, 31))
GROUPS = {
    "g_levels": (["rolling_7_sales", "rolling_14_sales"], "ha_"),
    "g_lags": (["lag_7_sales", "lag_14_sales", "lag_21_sales"], "hb_"),
    "g_fe": (["lag_1_fe", "lag_7_fe", "rolling_7_fe"], "hc_"),
    "g_wd": (["same_wd_mean4", "same_wd_max4"], "hd_"),
    "g_fe_wd": (["lag_1_fe", "lag_7_fe", "rolling_7_fe", "same_wd_mean4", "same_wd_max4"], "he_"),  # added after Kevin's question: the two groups that kept the outside gain, together
}


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity_fixed(frame, DATE_COL)
    jan = s82.in_window(out[DATE_COL], *JAN)
    dec = s82.in_window(out[DATE_COL], *DEC)
    new = {}
    for members, prefix in GROUPS.values():
        for c in INPUTS:
            col = out[c].astype(float).to_numpy().copy()
            col[jan] = np.nan
            if c in members:
                col[dec] = np.nan
            new[prefix + c] = col
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def piece_features(piece: str, variant: str) -> list[str]:
    prefix = GROUPS[variant][1]
    features = s54.piece_features(piece, "none") + ([pi.COLUMN_FIXED] if piece == "Saturday" else [])
    return [prefix + f if f in INPUTS else f for f in features]


def init_worker84() -> None:
    s47.init_worker47()
    clash = {g[1] + c for g in GROUPS.values() for c in INPUTS} & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for variant in GROUPS:
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
    window = s82.in_window(days["date"], (12, 26), (1, 8))
    done, done81, done82 = read_done(RESULTS), read_done(s81.RESULTS), read_done(s82.RESULTS)
    pd.set_option("display.width", 250)

    def package(source: dict, variant: str) -> np.ndarray:
        p = {name: mine(source, name, variant) for name in PIECES}
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base = package(done81, "base")
    preds = {"j1_8": package(done82, "j1_8"), "d26_j8 (all ten, 26 Dec-8 Jan)": package(done82, "d26_j8")} | {g: package(done, g) for g in GROUPS}
    rows = {}
    for name, p in preds.items():
        diff = np.abs(p - a) - np.abs(base - a)
        rows[name] = {"net, 585 days": float(diff.mean()), f"the {int(window.sum())} days 26 Dec-8 Jan": float(diff[window].mean()), "all other days": float(diff[~window].mean())}
    table = pd.DataFrame(rows).T
    print("EACH VARIANT MINUS THE UNMASKED PACKAGE, the same 25 seeds (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    ref_net, ref_out = float(table.loc["j1_8", "net, 585 days"]), float(table.loc["j1_8", "all other days"])
    print(f"\nj1_8: net {ref_net:+.2f}, outside {ref_out:+.2f}")
    for g in GROUPS:
        net, out = float(table.loc[g, "net, 585 days"]), float(table.loc[g, "all other days"])
        verdict = "CAPTURES BOTH" if (net <= ref_net - 2.0 and out <= ref_out + 1.0) else "COSTS" if net > ref_net + 2.0 else "NO GAIN over 1 to 8 January" if abs(net - ref_net) <= 2.0 else "IN BETWEEN (better net, but the outside gain is lost)"
        print(f"READING SET BEFOREHAND, {g}: net {net:+.2f} ({net - ref_net:+.2f} against j1_8), outside {out:+.2f}: {verdict}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "subsets.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(GROUPS)} groups x {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds)")
        s47.init_worker47()
        frame = add_columns(add_hydro_thu(_STATE["df"]))
        dt = frame[DATE_COL]
        jan, dec = s82.in_window(dt, *JAN), s82.in_window(dt, *DEC)
        for variant, (members, prefix) in GROUPS.items():
            for c in INPUTS:
                col = frame[prefix + c]
                expect = jan | (dec if c in members else np.zeros(len(frame), dtype=bool))
                assert col[expect].isna().all() and np.array_equal(col[~expect].to_numpy(), frame.loc[~expect, c].to_numpy(), equal_nan=True), f"{variant}: {c}'s mask is wrong"
            print(f"{variant:9s} masks {members} on 26-31 December ({int(dec.sum())} rows) and all ten on 1-8 January ({int(jan.sum())} rows)")
            for piece in PIECES:
                f = piece_features(piece, variant)
                print(f"   {piece:26s} {sum(x.startswith(prefix) for x in f)} masked inputs, of which {sum(x.startswith(prefix) and x[len(prefix):] in members for x in f)} also masked in late December")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker84, "series84")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
