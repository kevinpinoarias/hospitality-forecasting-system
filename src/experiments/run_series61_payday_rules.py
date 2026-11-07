"""
Series 61 - which payday dates and which window: the last working day, the 25th, the 28th and the 15th, tried together and one at a time.

Why. The model's payday columns assume one rule (payday = the last working day of the month). The CIPP Payslip Statistics Survey 2023 (payroll teams,
UK) puts the last working day at 34% of monthly payrolls and the 25th and the 28th at about 30% each, the 15th at 18% (a respondent can pick several
dates; these are shares of payrolls, not of employees). ONS ASHE has 75-78% of employee jobs paid monthly and the rest weekly, fortnightly or
four-weekly. No Scottish split or hospitality split was found.

What the deployed model's own backtest errors show (post model before the correction, 585 days, actual minus predicted; a descriptive look, written
down BEFORE this series was built):
  by day of month    days 1-24 flat; Fri-Sun on the 25th +837 and the 28th +1,035, the 29th +508 (Fri-Sun) and +499 (Mon-Thu); each about 1.5-2 standard errors
  by days since the latest of the three paydays (last working day, 25th, 28th; a date on a weekend or bank holiday is paid on the working day before)
                     Fri-Sun: 0 days +377, 1 day +810 (n=25, 2.9 standard errors), 2 days -179, 3 days -569, 4 days -680
                     Mon-Thu: 1 day +361, otherwise near zero
                     the Fri-Sun days at 0-1 days are 23 paydays that fall on a Friday (+377) and the 23 Saturdays right after them (+912)
  single rules       the day after payday, Fri-Sun: 28th +1,313, last working day +708, 25th +490, 15th +112 (nothing)
The reading is therefore a window of the payday and ONE day after it, for the three month-end rules, with the 15th as the control.

CAUTION: these patterns were found on the same 585 backtest days this series is scored on, so the backtest gain is optimistic by construction. The
only days that were not seen are the 56 from 7 September 2026, which hold three Friday paydays and their Saturdays (25-26 September, 23-24 October and
30-31 October). Read the result with that in mind, and prefer folds-better and the direction on weekends over the size of the gain.

Variants. Each is added next to the deployed columns (kept as they are, flaw included) and run on its own, never in combination:

  mix_since_capped      days since the latest payday of the three month-end rules (last working day, 25th, 28th), capped at 2 (0, 1, 2 or more)
  mix_window01          1 on the payday and the day after it, for any of the three; 0 otherwise
  friday_payday_weekend 1 on a Friday that is a payday under any of the three, and on the Saturday after it
  mix_signed_distance   signed days to the nearest payday of the three (negative before, positive after), clipped to -7..+7
  rule_25_window01      1 on the 25th-rule payday and the day after it
  rule_28_window01      1 on the 28th-rule payday and the day after it
  rule_15_window01      1 on the 15th-rule payday and the day after it (the control: nothing expected)

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval; where it comes from by weekday and piece (Saturday above all); in how many folds it is better; the new days
  expected   a modest gain on Saturday and Friday from the mix variants, nothing from the 15th control; if the 15th control gains as much as the
             others, the gain is not about payday dates
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted; the live API builds the same columns, so a
             new column would change the serving code too

Run:  python -m src.experiments.run_series61_payday_rules list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series61"
RESULTS = OUT_DIR / "series61_payday_rules.jsonl"
VARIANTS = ["none", "mix_since_capped", "mix_window01", "friday_payday_weekend", "mix_signed_distance",
            "rule_25_window01", "rule_28_window01", "rule_15_window01"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = VARIANTS[1:]
MIX_RULES = ["last_working_day", 25, 28]  # the three month-end rules; the 15th is tried on its own as the control


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    return cols + ([variant] if variant != "none" else [])


def month_payday(period: pd.Period, rule, holiday_calendar) -> pd.Timestamp:
    """The payday of one calendar month: the date, or the working day before it when it falls on a weekend or bank holiday."""
    day = period.to_timestamp(how="end").normalize() if rule == "last_working_day" else pd.Timestamp(year=period.year, month=period.month, day=rule)
    while day.weekday() >= 5 or day in holiday_calendar:
        day -= pd.Timedelta(days=1)
    return day


def days_around(dates: pd.Series, rule, holiday_calendar) -> tuple[np.ndarray, np.ndarray]:
    """Days since the latest payday on or before each date, and days to the next payday on or after it, for one rule."""
    d = dates.dt.normalize()
    periods = pd.period_range(d.min().to_period("M") - 2, d.max().to_period("M") + 2, freq="M")
    paydays = np.sort(np.array([month_payday(p, rule, holiday_calendar) for p in periods], dtype="datetime64[ns]"))
    values = d.to_numpy().astype("datetime64[ns]")
    latest = paydays[np.searchsorted(paydays, values, side="right") - 1]
    upcoming = paydays[np.searchsorted(paydays, values, side="left")]
    day = np.timedelta64(1, "D")
    return ((values - latest) // day).astype(int), ((upcoming - values) // day).astype(int)


def add_payday_rule_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    dates, dow = out[DATE_COL], out[DATE_COL].dt.dayofweek.to_numpy()
    since, to = zip(*[days_around(dates, rule, calendar) for rule in MIX_RULES])
    since_mix, to_mix = np.min(since, axis=0), np.min(to, axis=0)
    out["mix_since_capped"] = np.minimum(since_mix, 2)
    out["mix_window01"] = (since_mix <= 1).astype(int)
    out["friday_payday_weekend"] = (((dow == 4) & (since_mix == 0)) | ((dow == 5) & (since_mix == 1))).astype(int)
    out["mix_signed_distance"] = np.clip(np.where(since_mix <= to_mix, since_mix, -to_mix), -7, 7)
    for rule in (25, 28, 15):
        rule_since, _ = days_around(dates, rule, calendar)
        out[f"rule_{rule}_window01"] = (rule_since <= 1).astype(int)
    return out


def init_worker61() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_payday_rule_columns(add_hydro_thu(_STATE["df"]))


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
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, each variant {len(piece_features(piece, VARIANTS[1]))}")
        from src.features.build_features import add_payday_features

        df, _ = load_protocol()
        df = add_payday_rule_columns(df)
        calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
        lwd_since, _ = days_around(df[DATE_COL], "last_working_day", calendar)
        deployed = add_payday_features(df[[DATE_COL]])
        assert ((deployed["is_payday"] == 1) == (lwd_since == 0)).all(), "the last-working-day rule here is not the deployed payday"
        for name in NEW_COLUMNS:
            print(f"{name:24s} on {int((df[name] != 0).sum()):4d} of {len(df)} days; values {sorted(df[name].unique().tolist())[:15]}")
        fw = df[df["friday_payday_weekend"] == 1]
        print("friday_payday_weekend by weekday:", fw[DATE_COL].dt.day_name().value_counts().to_dict())
        new = df[df[DATE_COL] >= CUTOFF]
        print(f"the {len(new)} new days: mix_window01 on {int(new['mix_window01'].sum())}, friday_payday_weekend on {int(new['friday_payday_weekend'].sum())}: "
              f"{[str(x.date()) for x in new.loc[new['friday_payday_weekend'] == 1, DATE_COL]]}")
        show = df[df[DATE_COL].between("2025-09-22", "2025-10-02")]
        print(show[[DATE_COL] + NEW_COLUMNS].to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker61, "series61")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
