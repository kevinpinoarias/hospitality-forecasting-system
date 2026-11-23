"""
Series 70 - payday intensity: one number for how much pay is arriving around a day, added up over the pay dates in use.

Series 61 and 64 tried the pay dates (the last working day, the 25th, the 28th, the 15th) as flags, windows and a distance to the nearest payday. Those take the NEAREST payday
and ignore the others, so a day near two or three paydays at once looks the same as a day near one, and every rule counts equally. Intensity adds the rules up instead: each pay rule adds a
bump on the days around its payday, a more common rule adds a bigger bump, and the day gets the total. A day at the end of the month, where the last working day, the 28th and
the 25th crowd together, gets more than a day near the 15th alone.

The weights are the shares of payroll teams that use each date in the CIPP Payslip Statistics Survey 2023 (UK, multi-select, no stated sample size): last working day 0.34, 25th 0.30, 28th 0.30,
15th 0.18. They are rough: a respondent could pick several dates and no Scottish or hospitality split exists. Each rule's payday is that day of the month, or the working day before it when it falls
on a weekend or a Scottish bank holiday (`days_around` and `month_payday` of Series 61).

Variants, each ONE column added on top of the deployed 38 (flaw included) and run on its own. The bump of one rule depends on the days since its latest payday (k), or for the symmetric shape on the
distance to its nearest payday (k):

  intensity_short       bump 1.0 on payday (k = 0), 0.5 the day after (k = 1), 0 after that   (the pattern the errors suggested in Series 61: the payday and the day after)
  intensity_long        1.0, 0.75, 0.5, 0.25 for k = 0 to 3, 0 after that                    (a four-day fade)
  intensity_sym         1 - k / 4 for the nearest payday at k = 0 to 3 days, either side       (a triangle, so the build-up before payday counts too)
  intensity_long_equal  as intensity_long with every rule weighted 1 (tests whether the survey weights matter)

Series 54's design: every variant in every piece of the weekly log (the post model's four pieces, the hydro Monday-to-Thursday piece, the raw
Saturday fallback), 100 seeds (200-299), 585 days and the 56 new days, the two models assembled and corrected, read by judgment on the net effect.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  look at    for each variant, the change in the post and hydro models' MAE after the correction with a 95% moving-block bootstrap interval; by weekday and
             piece (Friday and Saturday above all); folds better; the new days; and the equal-weight variant against the survey-weight one
  expected   nothing: it re-weights ground Series 61 and 64 covered, and no payday encoding has beaten the deployed columns. It is run because Kevin asked for the idea to be closed
  would matter  an interval below zero in both models, and the new days not worse
  limits     no pass/fail gate; nothing in production changes; the survey weights are rough; a new column would change the serving code too

Run:  python -m src.experiments.run_series70_payday_intensity list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.run_series61_payday_rules import add_payday_rule_columns, days_around
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series70"
RESULTS = OUT_DIR / "series70_payday_intensity.jsonl"
RULES = ["last_working_day", 25, 28, 15]
SHARES = {"last_working_day": 0.34, 25: 0.30, 28: 0.30, 15: 0.18}
EQUAL = {rule: 1.0 for rule in RULES}
FADES = {"short": [1.0, 0.5], "long": [1.0, 0.75, 0.5, 0.25]}
RECIPES = {  # variant -> (shape, weights)
    "intensity_short": ("short", SHARES),
    "intensity_long": ("long", SHARES),
    "intensity_sym": ("sym", SHARES),
    "intensity_long_equal": ("long", EQUAL),
}
VARIANTS = ["none", *RECIPES]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = list(RECIPES)


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    return cols + ([variant] if variant != "none" else [])


def bump(shape: str, since: np.ndarray, to: np.ndarray) -> np.ndarray:
    """One rule's bump from the days since its latest payday and the days to its next one."""
    if shape == "sym":
        return np.clip(1.0 - np.minimum(since, to) / 4.0, 0.0, None)
    fade = np.array(FADES[shape] + [0.0])
    return fade[np.minimum(since, len(fade) - 1)]


def add_intensity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    around = {rule: days_around(out[DATE_COL], rule, calendar) for rule in RULES}
    for name, (shape, weights) in RECIPES.items():
        out[name] = sum(weights[rule] * bump(shape, *around[rule]) for rule in RULES)
    return out


def init_worker70() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_intensity_columns(add_hydro_thu(_STATE["df"]))


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
        df, _ = load_protocol()
        df = add_intensity_columns(df)
        total = sum(SHARES.values())
        for name in NEW_COLUMNS:
            x = df[name]
            limit = total if "equal" not in name else float(len(RULES))
            assert (x >= 0).all() and (x <= limit + 1e-9).all(), f"{name} out of range"
            print(f"{name:22s} nonzero on {int((x > 0).sum()):4d} of {len(df)} days; max {x.max():.2f}; mean {x.mean():.3f}; share of days above 0.5: {100 * (x > 0.5).mean():.0f}%")
        rules = add_payday_rule_columns(df)
        print("correlation of intensity_long with mix_signed_distance (nearest payday) :", round(float(df["intensity_long"].corr(rules["mix_signed_distance"].abs())), 3))
        print("correlation of intensity_short with the deployed is_payday              :", round(float(df["intensity_short"].corr(df["is_payday"])), 3))
        show = df[df[DATE_COL].between("2025-09-12", "2025-10-02")]
        print(show[[DATE_COL, "intensity_short", "intensity_long", "intensity_sym", "intensity_long_equal"]].round(2).assign(
            weekday=show[DATE_COL].dt.strftime("%a")).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker70, "series70")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
