"""
Series 62 - is the deployed `days_since_payday` really a day-of-month ramp, and is that why it beats the repair?

Series 59 found that repairing the payday columns made the model worse (post model after the correction +4.87 pounds a day) and that dropping them
was worse still (+7.81). Series 60 and 61 found no explicit flag or window that helps. The untested explanation is this: the deployed
`days_since_payday` counts from the PREVIOUS month's payday and never resets inside the month, so across a calendar month it climbs from about 1 to
about 30 and stays near 30 on payday and on the days after an early payday. That is close to a plain day-of-month ramp (with a few days of offset
where the previous month's payday fell early). The repair resets it to 0 on payday, which breaks the ramp. If that is the reason, the ramp is what the
model uses, not the payday.

Three variants, each swapping columns in the same positions (column order is unchanged), each run on its own:

  since_as_dom       days_since_payday replaced by day_of_month (1..31). Everything else as deployed.
                     Tests the ramp reading directly: if this matches the deployed model, the column is working as a day-of-month ramp.
  repair_keep_ramp   days_to_payday and is_payday_window_pm3 repaired (Series 59's `_fixed` versions), days_since_payday left as deployed.
                     Splits Series 59's `repair_all`: shows whether the loss came from the reset in days_since_payday or from the other two.
  repair_dom_ramp    days_to_payday and is_payday_window_pm3 repaired, days_since_payday replaced by day_of_month. The cleanest "correct" family:
                     a true ramp plus true payday distances, no flaw anywhere.

Series 54 added day_of_month on top of the deployed columns and found nothing; this series REPLACES days_since_payday with it, which is a different test.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction against the deployed columns, with a
             95% moving-block bootstrap interval; by weekday and piece; folds better; the new days. Compare with Series 59: `repair_all` was
             +4.87 (post) and `fix_since_only` is in the Series 59 report.
  ramp holds if   since_as_dom is within about +-1 pound of the deployed model with an interval around zero
  reset hurts if  repair_keep_ramp is near the deployed model while Series 59's repair_all was worse
  neither if      since_as_dom is clearly worse than the deployed model: then the deployed column carries something the day of month does not
  expected   since_as_dom close to the deployed model; repair_dom_ramp close to it or slightly better
  limits     no pass/fail gate; nothing in production changes; the live API builds the same columns, so a swap would change the serving code too

Run:  python -m src.experiments.run_series62_payday_ramp list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series59_payday_repair as s59
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series62"
RESULTS = OUT_DIR / "series62_payday_ramp.jsonl"
VARIANTS = ["none", "since_as_dom", "repair_keep_ramp", "repair_dom_ramp"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = ["day_of_month"] + s59.NEW_COLUMNS
SWAPS = {
    "since_as_dom": {"days_since_payday": "day_of_month"},
    "repair_keep_ramp": {"days_to_payday": "days_to_payday_fixed", "is_payday_window_pm3": "is_payday_window_pm3_fixed"},
    "repair_dom_ramp": {"days_since_payday": "day_of_month", "days_to_payday": "days_to_payday_fixed",
                        "is_payday_window_pm3": "is_payday_window_pm3_fixed"},
}


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    return [SWAPS.get(variant, {}).get(c, c) for c in cols]


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return s54.add_day_of_month_columns(s59.add_repaired_columns(frame))


def init_worker62() -> None:
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
                assert len(piece_features(piece, v)) == len(piece_features(piece, "none")), "a swap changed the number of columns"
                assert all(c in set(piece_features(piece, v)) for c in SWAPS[v].values()), "a swapped-in column is missing"
        df, _ = load_protocol()
        df = add_columns(df)
        gap = df["days_since_payday"] - df["day_of_month"]
        print(f"\n{len(df)} days. days_since_payday against day_of_month: correlation {df['days_since_payday'].corr(df['day_of_month']):.3f}; "
              f"difference (days_since_payday minus day_of_month) ranges {int(gap.min())} to {int(gap.max())}, "
              f"within 0-5 on {int(gap.between(0, 5).sum())} days; outside on {int((~gap.between(0, 5)).sum())}")
        print("difference counts:", gap.value_counts().sort_index().to_dict())
        show = df[df[DATE_COL].between("2025-09-26", "2025-10-03")]
        print(show[[DATE_COL, "day_of_month", "days_since_payday", "days_since_payday_fixed", "days_to_payday", "days_to_payday_fixed"]].to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker62, "series62")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
