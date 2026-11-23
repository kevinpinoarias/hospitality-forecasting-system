"""
Series 33 - the first-week correction, forward-chained on all six breaks, pooled against per-type and median.

Series 32 found the first week of the Easter and summer breaks is under-forecast in all six breaks of the data (2024 included, from
forecasts made without a window around each break) and that a correction learned from other breaks helps. Its realistic test
(only earlier breaks) had three scorable breaks and the summer-2026 correction overshot, because the learned shift was dominated by the
record summer of 2025. Here the correction is forward-chained on the leave-one-break-out forecasts of Series 32, which cover all six
breaks including 2024, so the correction itself only ever learns from EARLIER breaks (the forecasts it corrects come from a model that
saw later data, which is the optimism to remember).

Three variants, each additive, strength 0.5, first week only (the first 7 days of the break):
  pooled mean     the shift is the mean of (forecast minus actual) over the first weeks of ALL earlier Easter and summer breaks
  per type mean   the same, but only from earlier breaks of the SAME kind (Easter from Easters, summer from summers)
  pooled median   like pooled, but the shift is the median over those days, so one record day cannot dominate it

PRE-REGISTERED PLAN (written before any result was seen)
  scored      the four breaks every variant can correct: Easter 2025, summer 2025, Easter 2026, summer 2026 (28 days). The pooled
              variants can also correct summer 2024 (one earlier break, Easter 2024); that is reported as context only
  measure     the change in MAE on the scored days (after minus before, so negative is better) and the number of breaks that improve,
              NOT the level of MAE. A 95% interval from resampling whole breaks is reported but four breaks cannot clear a bar
  choice      the variant with the largest improvement on the scored days is the one pre-registered for the Easter and summer 2027 breaks,
              provided it improves at least three of the four breaks; if none does, the 2027 rule of Series 32 stands (pooled mean)
  context     the learned shift against the actual bias of each break, to see where a variant over- or under-corrects

Run:  python -m src.experiments.run_series33_first_week_by_type
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series29_school_calendar as s29
from src.experiments import run_series32_first_week_correction as s32
from src.experiments.series19_21_common import SUFFIX, load_protocol
from src.models.final_model import DATE_COL

REPORT_DIR = CONFIG.path("experiments_dir") / f"series33_first_week_by_type{SUFFIX}"
STRENGTH = 0.5
VARIANTS = {"pooled mean": ("pooled", "mean"), "per type mean": ("type", "mean"), "pooled median": ("pooled", "median")}
SCORED = ["Easter 2025", "Summer 2025", "Easter 2026", "Summer 2026"]


def kind(name: str) -> str:
    return name.split()[0].lower()


def first_week_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The seven first-week days of every break (the leave-one-break-out frame also holds the last week of term)."""
    return frame[frame["date"] >= frame["start"]].copy()


def learned_shift(pred: np.ndarray, actual: np.ndarray, how: str) -> float:
    gap = pred - actual
    return float(np.median(gap) if how == "median" else np.mean(gap))


def forward_chain(first: pd.DataFrame, history: str, how: str, strength: float = STRENGTH) -> pd.DataFrame:
    """Correct each break's first week from the first weeks of EARLIER breaks (all of them, or only the same kind). One row per break
    that has history, with the learned shift and the MAE before and after."""
    order = first.drop_duplicates("break").sort_values("start")["break"].tolist()
    rows = []
    for i, name in enumerate(order):
        earlier = [b for b in order[:i] if history == "pooled" or kind(b) == kind(name)]
        if not earlier:
            continue
        past = first[first["break"].isin(earlier)]
        now = first[first["break"] == name]
        shift = learned_shift(past["pred"].to_numpy(), past["actual"].to_numpy(), how)
        corrected = now["pred"].to_numpy() - strength * shift
        a, p = now["actual"].to_numpy(), now["pred"].to_numpy()
        rows.append({"break": name, "earlier_breaks": len(earlier), "days": len(now), "learned_shift": shift, "actual_bias": float((p - a).mean()),
                     "before_MAE": float(np.abs(p - a).mean()), "after_MAE": float(np.abs(corrected - a).mean()),
                     "change": float(np.abs(corrected - a).mean() - np.abs(p - a).mean())})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame, names: list[str]) -> dict:
    sub = table[table["break"].isin(names)]
    days = sub["days"].to_numpy().astype(float)
    mean, lo, hi = s32.cluster_bootstrap(sub["change"].to_numpy(), days)
    return {"breaks_scored": len(sub), "days": int(days.sum()),
            "before_MAE": float((sub["before_MAE"] * days).sum() / days.sum()), "after_MAE": float((sub["after_MAE"] * days).sum() / days.sum()),
            "change": mean, "lo": lo, "hi": hi, "breaks_improved": int((sub["change"] < 0).sum())}


def main() -> None:
    df, _ = load_protocol()
    df = s29.with_calendar(df)
    breaks = s32.breaks_in_data(df[DATE_COL])
    first = first_week_rows(s32.lobo_frame(df, breaks))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    print("First-week bias of the final model from leave-one-break-out forecasts (forecast minus actual):\n",
          first.groupby("break", sort=False).apply(lambda g: pd.Series({"MAE": (g["pred"] - g["actual"]).abs().mean(), "bias": (g["pred"] - g["actual"]).mean()}),
                                                   include_groups=False).round(0).to_string())
    results, per_break = [], {}
    for label, (history, how) in VARIANTS.items():
        table = forward_chain(first, history, how)
        per_break[label] = table
        table.assign(variant=label).to_csv(REPORT_DIR / f"by_break_{label.replace(' ', '_')}.csv", index=False)
        results.append({"variant": label, **summary(table, SCORED)})
        print(f"\n{label}: break by break (strength {STRENGTH}, first week only):\n", table.round(0).to_string(index=False))
    res = pd.DataFrame(results)
    res.to_csv(REPORT_DIR / "summary_scored_breaks.csv", index=False)
    print("\nThe four breaks every variant can correct (28 days). Change = after minus before, £ (negative is better); interval from resampling whole breaks:\n",
          res.round(1).to_string(index=False))
    context = per_break["pooled mean"]
    c = context[context["break"] == "Summer 2024"]
    print("\nContext: summer 2024, pooled variants only (one earlier break, Easter 2024):\n", c.round(0).to_string(index=False))

    best = res.sort_values("change").iloc[0]
    ok = int(best["breaks_improved"]) >= 3
    print(f"\nLargest improvement: {best['variant']} ({best['change']:+.1f} on {int(best['days'])} days, {int(best['breaks_improved'])} of {int(best['breaks_scored'])} breaks improve). "
          f"{'Pre-registered for the 2027 breaks.' if ok else 'Fewer than three breaks improve: the Series 32 rule (pooled mean) stands for 2027.'}")
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
