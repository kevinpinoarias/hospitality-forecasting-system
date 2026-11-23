"""
Series 63 - is the 0-to-3-day offset what makes the deployed `days_since_payday` work?

Series 62 showed the deployed `days_since_payday` is the day of the month plus an offset of 0 to 3 days on all 1,006 days (the number of days
between the previous month's payday and that month's end: 0 when the last working day was also the last calendar day, up to 3 when the month ended
on a weekend or bank holiday). Replacing the column by plain `day_of_month` cost 2.98 pounds a day (interval 0.69 to 5.18, better in 1 of 10 folds), so the
column is not just a day-of-month ramp. The offset is the obvious carrier of what is left. This series separates it.

  prev_payday_gap  = deployed days_since_payday minus day_of_month (0 to 3). It is the same on every day of a calendar month, so it describes
                     "how early did last month's payday fall" and nothing else.

Two variants, each run on its own:

  dom_plus_gap     days_since_payday replaced by TWO columns in its place: day_of_month, then prev_payday_gap. The model sees the same information as
                   the deployed column, but split apart, so a tree can use the ramp and the gap separately. One column more than deployed.
  deployed_plus_gap the deployed columns as they are, with prev_payday_gap added at the end of the piece's inputs.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 new days from 7 September 2026
  look at    for each variant, the change in the post and hydro models' MAE after the correction against the deployed columns, with a 95%
             moving-block bootstrap interval; by weekday and piece; folds better; the new days
  gap is the signal if   dom_plus_gap is close to the deployed model (within about 1 pound, interval around zero), and Series 62's since_as_dom was not
  gap is not needed if   dom_plus_gap is as bad as since_as_dom: the loss then comes from the combined column, not the offset on its own
  extra information if   deployed_plus_gap improves on the deployed model (it should not: the gap is already inside days_since_payday)
  limits     no pass/fail gate; nothing in production changes; the live API builds the same columns

Run:  python -m src.experiments.run_series63_payday_gap list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series63"
RESULTS = OUT_DIR / "series63_payday_gap.jsonl"
VARIANTS = ["none", "dom_plus_gap", "deployed_plus_gap"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = ["day_of_month", "prev_payday_gap"]


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    if variant == "dom_plus_gap":
        out = []
        for c in cols:
            out += ["day_of_month", "prev_payday_gap"] if c == "days_since_payday" else [c]
        return out
    if variant == "deployed_plus_gap":
        return cols + ["prev_payday_gap"]
    return cols


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = s54.add_day_of_month_columns(frame)
    out["prev_payday_gap"] = out["days_since_payday"] - out["day_of_month"]
    return out


def init_worker63() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in VARIANTS:
            if variant == "none":
                continue  # every baseline exists already: Series 46-47 for the post pieces, Series 54's `none` for the other two
            features = piece_features(piece, variant)
            for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS), size):
                    chunk = SEEDS[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant,
                                "seeds": chunk, "features": features})
    return out


def report() -> None:
    s54.report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=s54.RESULTS)


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, " + ", ".join(
                f"{v} {len(piece_features(piece, v))}" for v in VARIANTS[1:]))
            for v in VARIANTS[1:]:
                feats = piece_features(piece, v)
                assert len(feats) == len(piece_features(piece, "none")) + 1, "each variant should add exactly one column"
                assert "prev_payday_gap" in feats
            assert "days_since_payday" not in piece_features(piece, "dom_plus_gap")
        df, _ = load_protocol()
        df = add_columns(df)
        gap = df["prev_payday_gap"]
        print(f"\n{len(df)} days; gap values {gap.value_counts().sort_index().to_dict()}")
        assert gap.between(0, 3).all()
        by_month = df.groupby(df[DATE_COL].dt.to_period("M"))["prev_payday_gap"].nunique()
        print("months where the gap is not constant:", int((by_month > 1).sum()), "of", len(by_month))
        print(df.loc[df[DATE_COL].between("2025-09-28", "2025-10-03"), [DATE_COL, "days_since_payday", "day_of_month", "prev_payday_gap"]].to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker63, "series63")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
