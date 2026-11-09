"""
Series 82 - where exactly the recent-sales inputs should be switched off around the new year.

Series 80 switched ten recent-sales inputs off for 1 to 15 January (in the training rows and the forecast days) and improved the package by 6.6 pounds a day over the 585 days; Series 81 (25 seeds) reproduced it (-6.08) and ruled out
the zero-sales closed days as the cause. The window 1 to 15 January was a guess. The inputs reach back different distances: the 7-day average clears Christmas by about 8 January, the 14-day one by about 15 January, the
"same weekday over the last four weeks" inputs by about 29 January; and the days after Christmas (26 to 31 December) have the same problem. This series moves the two ends.

Six windows, each refitted in all four pieces of the current package (Monday-Thursday with the hydro flag, Friday, Saturday with `intensity_fixed`, Sunday), the ten inputs of Series 80/81 set to missing on every row dated inside
the window, in the training rows and the forecast days alike (the original pipeline inputs, masked as Series 80 did):
  j1_8     1 to 8 January            j1_22    1 to 22 January           j1_29    1 to 29 January
  d26_j8   26 December to 8 January  d26_j15  26 December to 15 January d26_j22  26 December to 22 January
The references are Series 81's own fits on the same seeds: `base` (unmasked) and `jan_mask` (1 to 15 January, the window every variant here is compared with).

Protocol: 25-seed bagged averages (seeds 200-224), 10 folds, 585 days, the deployed weekend correction; 6 windows x 4 pieces x 5 tasks of 5 seeds = 120 tasks.

READING (set before any result was seen)
  measure     the NET change of the post package over all 585 days against `base` (pounds a day; negative = better), with the change over the days inside the window shown beside it
  best        the window with the most negative net change; any window within 1.0 of it is TIED with it (25 seeds: a difference of 2 or less is within the noise)
  where       the stop date: if the net change stops growing past a date (a later stop is within 1.0 of an earlier one), the earlier one is enough; the start date: 26 December helps if each d26_ window is at least 1.0
              better than its 1-January twin (d26_j15 against jan_mask; d26_j8 against j1_8; d26_j22 against j1_22)
  expected    the gain grows from the 8th to the 15th to the 22nd and flattens (the longer the window, the more of the lags' reach it covers), and starting on 26 December adds a little; the longest windows are the most
              likely to start costing in the windows' own days (the inputs are informative again once Christmas has left them)
  limits      25 seeds; one January in the backtest; the same 585 days; the window is fixed by the best of six tries on the same days (a mild selection effect), so the chosen window is a candidate for a rule written
              down beforehand and read on January 2027, not a result; a refit would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series82_mask_windows list | run [--workers 10] | report
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
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel, test_days, read_done
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series82"
RESULTS = OUT_DIR / "series82_mask_windows.jsonl"
SEEDS25, PIECES, INPUTS = s81.SEEDS25, s81.PIECES, s81.INPUTS
# variant -> (first (month, day), last (month, day), column prefix); a window that starts in December wraps over the new year
WINDOWS = {
    "j1_8": ((1, 1), (1, 8), "wa_"), "j1_22": ((1, 1), (1, 22), "wb_"), "j1_29": ((1, 1), (1, 29), "wc_"),
    "d26_j8": ((12, 26), (1, 8), "wd_"), "d26_j15": ((12, 26), (1, 15), "we_"), "d26_j22": ((12, 26), (1, 22), "wf_"),
}


def in_window(dates: pd.Series, first: tuple[int, int], last: tuple[int, int]) -> np.ndarray:
    d = pd.to_datetime(dates)
    md = (d.dt.month * 100 + d.dt.day).to_numpy()
    a, b = first[0] * 100 + first[1], last[0] * 100 + last[1]
    return (md >= a) | (md <= b) if a > b else (md >= a) & (md <= b)


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity_fixed(frame, DATE_COL)
    new = {}
    for first, last, prefix in WINDOWS.values():
        w = in_window(out[DATE_COL], first, last)
        for c in INPUTS:
            col = out[c].astype(float).to_numpy().copy()
            col[w] = np.nan
            new[prefix + c] = col
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def piece_features(piece: str, variant: str) -> list[str]:
    prefix = WINDOWS[variant][2]
    features = s54.piece_features(piece, "none") + ([pi.COLUMN_FIXED] if piece == "Saturday" else [])
    return [prefix + f if f in INPUTS else f for f in features]


def init_worker82() -> None:
    s47.init_worker47()
    clash = {w[2] + c for w in WINDOWS.values() for c in INPUTS} & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for variant in WINDOWS:
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
    done, done81 = read_done(RESULTS), read_done(s81.RESULTS)
    pd.set_option("display.width", 250)

    def package(source: dict, variant: str) -> np.ndarray:
        p = {name: mine(source, name, variant) for name in PIECES}
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base = package(done81, "base")
    preds = {"j1_15 (Series 81's jan_mask)": (package(done81, "jan_mask"), ((1, 1), (1, 15)))} | {v: (package(done, v), (w[0], w[1])) for v, w in WINDOWS.items()}
    rows = {}
    for name, (pred, (first, last)) in preds.items():
        diff = np.abs(pred - a) - np.abs(base - a)
        inside = in_window(days["date"], first, last)
        rows[name] = {"window": f"{first[1]:02d}/{first[0]:02d} to {last[1]:02d}/{last[0]:02d}", "test days in it": int(inside.sum()), "net, 585 days": float(diff.mean()), "inside the window": float(diff[inside].mean()), "outside the window": float(diff[~inside].mean())}
    table = pd.DataFrame(rows).T
    order = ["j1_8", "j1_15 (Series 81's jan_mask)", "j1_22", "j1_29", "d26_j8", "d26_j15", "d26_j22"]
    table = table.loc[order]
    for c in ("net, 585 days", "inside the window", "outside the window"):
        table[c] = table[c].astype(float)
    print("EACH WINDOW MINUS THE UNMASKED PACKAGE, the same 25 seeds (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    nets = table["net, 585 days"]
    best = float(nets.min())
    tied = [n for n in nets.index if nets[n] <= best + 1.0]
    print(f"\nbest net change {best:+.2f}: {nets.idxmin()}; within 1.0 of it (tied): {tied}")
    print("stop date, starting on 1 January:", {k: round(float(nets[k]), 2) for k in ("j1_8", "j1_15 (Series 81's jan_mask)", "j1_22", "j1_29")})
    for early, twin in (("d26_j8", "j1_8"), ("d26_j15", "j1_15 (Series 81's jan_mask)"), ("d26_j22", "j1_22")):
        print(f"starting on 26 December: {early} {nets[early]:+.2f} against {twin} {nets[twin]:+.2f}: {nets[early] - nets[twin]:+.2f} ({'helps (at least 1.0 better)' if nets[early] - nets[twin] <= -1.0 else 'no clear difference' if abs(nets[early] - nets[twin]) < 1.0 else 'costs'})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "windows.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(WINDOWS)} windows x {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds)")
        s47.init_worker47()
        frame = add_columns(add_hydro_thu(_STATE["df"]))
        dt = frame[DATE_COL]
        for variant, (first, last, prefix) in WINDOWS.items():
            w = in_window(dt, first, last)
            for c in INPUTS:
                assert frame.loc[w, prefix + c].isna().all() and np.array_equal(frame.loc[~w, prefix + c].to_numpy(), frame.loc[~w, c].to_numpy(), equal_nan=True), f"{variant}: {c}'s mask is wrong"
            print(f"{variant:8s} masks {int(w.sum()):3d} rows ({dt[w].min():%d %b %Y} to {dt[w].max():%d %b %Y}); by year: {dt[w].dt.year.value_counts().sort_index().to_dict()}")
        assert in_window(pd.Series(pd.to_datetime(["2025-12-25", "2025-12-26", "2025-12-31", "2026-01-01", "2026-01-08", "2026-01-09"])), (12, 26), (1, 8)).tolist() == [False, True, True, True, True, False]
        print("the wrapping window includes 26 December to 8 January and nothing else")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker82, "series82")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
