"""
Series 94 - the exact form of the year-ago input of v1.14.

The version plan (docs/JANUARY_DECEMBER_CANDIDATES_2026_11.md): v1.14 is v1.13 plus a year-ago sales input in the Monday-Thursday (hydro), Friday and Saturday pieces (Series 93: the Sunday piece stays v1.13's). Series 90/92/93
used `sales_364`, the sales on the same weekday 364 days earlier: about -8 to -10 pounds a day against v1.13 on Monday-Saturday. Kevin (9 November 2026): test the exact form, for instance the average around last year's date,
and any other candidates.

The forms (each a new input, or two, added to the three pieces; every value comes from the full daily sales series and is known at forecast time; a row without the full history has none):
  A  `sales_364`        the sales on the same weekday 364 days earlier (the reference: Series 90's fits on the same seeds, reused)
  B  `sales_365`        the sales on the same calendar date 365 days earlier (the date, not the weekday, lined up)
  C  `ya_wk3`           the average of the same weekday 357, 364 and 371 days earlier (three weeks around last year's date, the weekday kept)
  D  `ya_c7`            the average of the 7 days centred on last year's date, 361 to 367 days earlier (the level around last year's date, weekday-agnostic)
  E  `ya_ratio`         `sales_364` divided by `ya_c7`: how last year's day compared with its own week
  F  `sales_364` + `ya_c7`   both inputs together
Fits: B-F x the Monday-Thursday (hydro), Friday and Saturday pieces x 5 tasks of 5 seeds (seeds 200-224) = 75 tasks, every one served with the S6 stand-in (models fitted on the real training rows, only the test rows'
`forecast_sales` differs, history from the training rows only), 10 folds, 585 days, the deployed weekend correction. The Sunday piece is v1.13's own (Series 89's unmasked fit) in every package, and v1.13's specialist routing
(the companion on 26-31 December, the masked model on 1-8 January) is applied in every package, as in Series 93.

READING (set before any result was seen)
  measure     the net change of each form's package against V13 (v1.13: the unmasked pieces outside the two windows) over the 585 days, pounds a day, negative = better, with 26-31 Dec, 1-8 Jan, all other days and the
              weekdays beside it; the forms are compared with the reference A on the SAME seeds
  better      a form is BETTER than A if its net change is at least 2.0 below A's (25-seed noise is about 2); TIED if within 2.0; WORSE if more than 2.0 above
  helps       a form HELPS if its net change against V13 is at most -2.0
  next        a form that is better than A or tied with a simpler build goes on to 100 and fresh seeds; nothing is decided on 25 seeds
  limits      25 seeds; the same 585 days (one December, one January); the gain of this input is uneven over the year (winter and summer better, early 2025, when the training rows had no year-ago values, and spring
              2026 worse), and the forms with a longer reach (B, C, D, E, F need up to 371 days of history) have slightly fewer rows with a value; the specialist models are not refitted with the input; the exact form
              is then a registration choice, Kevin's, with an amendment and a refit first

Run:  python -m src.experiments.run_series94_year_ago_forms list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series81_closed_days_in_lags as s81
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series89_standin_repeat_hybrid as s89
from src.experiments import run_series90_year_end_candidates as s90
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series94"
RESULTS = OUT_DIR / "series94_year_ago_forms.jsonl"
SEEDS25, PIECES = s81.SEEDS25, s90.PIECES
MON_SAT = ["Monday-Thursday (hydro)", "Friday", "Saturday"]
FORMS = {"B": ["sales_365"], "C": ["ya_wk3"], "D": ["ya_c7"], "E": ["ya_ratio"], "F": ["sales_364", "ya_c7"]}
LABELS = {"A": "A  sales_364 (the reference)", "B": "B  sales_365 (same date)", "C": "C  three-week same-weekday average", "D": "D  centred 7-day average", "E": "E  ratio of the day to its week", "F": "F  sales_364 + centred 7-day average"}


def init_worker94() -> None:
    s90.init_worker90()  # the frame of Series 90: the pieces' inputs, intensity_fixed, sales_364, days_since_xmas, the Christmas-break and mask columns
    df = _STATE["df"]
    sales, _ = s81.load_series()
    dates = pd.DatetimeIndex(df[DATE_COL])
    c7 = sales.shift(361).rolling(7).mean()
    wk3 = (sales.shift(357) + sales.shift(364) + sales.shift(371)) / 3.0
    ratio = (sales.shift(364) / c7).replace([np.inf, -np.inf], np.nan)
    new = {"sales_365": sales.shift(365).reindex(dates).to_numpy(), "ya_wk3": wk3.reindex(dates).to_numpy(), "ya_c7": c7.reindex(dates).to_numpy(), "ya_ratio": ratio.reindex(dates).to_numpy()}
    clash = set(new) & set(df.columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def features_for(form: str, piece: str) -> list[str]:
    return s90.base_features(piece) + FORMS[form]


def make_tasks() -> list[dict]:
    out = []
    for form in FORMS:
        for piece in MON_SAT:
            features = features_for(form, piece)
            for i in range(0, len(SEEDS25), s54.BACKTEST_CHUNK):
                chunk = SEEDS25[i:i + s54.BACKTEST_CHUNK]
                out.append({"id": f"backtest_{piece}_{form}94_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": f"{form}94", "seeds": chunk, "features": features})
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    jan8 = s82.in_window(days["date"], (1, 1), (1, 8))
    done94, done90, done89 = read_done(RESULTS), read_done(s90.RESULTS), read_done(s89.RESULTS)
    pd.set_option("display.width", 250)

    def package(p: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base = {pc: s90.s89_first25(done89, pc, "sd_base") for pc in PIECES}
    msk = {pc: s90.s89_first25(done89, pc, "sd_j1_8") for pc in PIECES}
    comp = {pc: s90.s89_first25(done89, pc, "sd_companion") for pc in PIECES}

    def routed(main: dict) -> dict:
        return {pc: np.where(six, comp[pc], np.where(jan8, msk[pc], main[pc])) for pc in PIECES}

    parent = package(routed(base))
    forms = {"A": {pc: (s90.mine(done90, pc, "B") if pc in MON_SAT else base[pc]) for pc in PIECES}}
    for f in FORMS:
        forms[f] = {pc: (s90.mine(done94, pc, f"{f}94") if pc in MON_SAT else base[pc]) for pc in PIECES}
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = {}
    for f, p in forms.items():
        d = np.abs(package(routed(p)) - a) - np.abs(parent - a)
        rows[LABELS[f]] = {"net, 585 days": float(d.mean()), "26-31 Dec": float(d[six].mean()), "1-8 Jan": float(d[jan8].mean()), "all other days": float(d[~(six | jan8)].mean()), **{n: float(d[dow == i].mean()) for i, n in enumerate(names)}}
    table = pd.DataFrame(rows).T
    print("EACH FORM OF THE YEAR-AGO INPUT (Monday-Saturday) MINUS V13, served with S6 (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    print(f"\nv1.13's own average miss on these seeds: {np.abs(parent - a).mean():.0f}\n")
    ref = float(table.iloc[0]["net, 585 days"])
    for f in FORMS:
        net = float(table.loc[LABELS[f], "net, 585 days"])
        verdict = "BETTER than A (at least 2.0 below)" if net <= ref - 2.0 else "TIED with A (within 2.0)" if net <= ref + 2.0 else "WORSE than A (more than 2.0 above)"
        print(f"READING SET BEFOREHAND, {LABELS[f]}: net {net:+.2f} ({net - ref:+.2f} against A): {verdict}; {'HELPS' if net <= -2.0 else 'does not clear -2.0'} against V13")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "forms.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(FORMS)} forms x {len(MON_SAT)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds, served with S6)")
        init_worker94()
        frame = _STATE["df"]
        sales, _ = s81.load_series()
        assert np.allclose(frame["sales_364"].to_numpy(), sales.shift(364).reindex(pd.DatetimeIndex(frame[DATE_COL])).to_numpy(), equal_nan=True)
        for col in ("sales_364", "sales_365", "ya_wk3", "ya_c7", "ya_ratio"):
            print(f"  {col:10s} has a value on {int(frame[col].notna().sum())} of {len(frame)} rows (first row with a value: {frame.loc[frame[col].notna(), DATE_COL].min():%d %b %Y})")
        for f in FORMS:
            missing = [x for x in features_for(f, "Friday") if x not in frame.columns]
            assert not missing, f"{f}: missing {missing}"
        d = pd.Timestamp("2026-01-05")
        row = frame[frame[DATE_COL] == d].iloc[0]
        print(f"on {d:%d %b %Y}: sales_364 {row['sales_364']:.0f}, sales_365 {row['sales_365']:.0f}, ya_wk3 {row['ya_wk3']:.0f}, ya_c7 {row['ya_c7']:.0f}, ya_ratio {row['ya_ratio']:.2f}")
    elif step == "run":
        run_parallel(make_tasks(), s90.fit_task90, RESULTS, workers, init_worker94, "series94")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
