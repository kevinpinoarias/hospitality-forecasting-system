"""
Series 65 - a payday-Saturday correction instead of a payday feature: split the Saturday correction in two, rather than stacking a second lift.

Series 59-64 tried to put the payday pattern into the model as columns. None of the encodings beat the deployed columns, and the
pattern itself rests on few days: of the 26 Saturdays that follow a Friday payday (payday = the last working day, the 25th or the 28th),
the deployed model under-predicted 20 (77%; 59% of other Saturdays are under-predicted). The three new days from 7 September 2026 went the other way.
A tree cannot learn from about 20 rows without paying for it elsewhere, which is probably why the features did not help.

The weekend correction in the final package (Series 24: half-life 8 weeks, strength 0.5, Saturday and Sunday) learns a bias from earlier blocks and
subtracts part of it, with no refit. It learns ONE bias for all Saturdays. Measured on folds 1-9 (before this series was run):

                                    days   lift from the existing correction   raw miss (actual - forecast)   miss left after it
    Saturdays after a Friday payday   22                 +221                            +907                       +686
    all other Saturdays               55                 +256                             +50                       -205

The existing correction lifts every Saturday by about the same amount: too little for the payday Saturdays, too much for the rest (the payday
Saturdays also pull the shared Saturday bias up, so the ordinary ones overshoot). This series tests two ways of giving the payday Saturdays their own treatment.
Both use the saved predictions of the deployed pieces, 100 seeds; nothing is refitted.

  split (PRIMARY)   Saturdays are two groups, each corrected from its own earlier blocks with the existing formula (half-life 8 weeks, strength 0.5,
                    recency weights, history from earlier folds only). The payday group needs 4 earlier payday Saturdays (the existing MIN_HISTORY);
                    until then it keeps the all-Saturday bias, as deployed. The other group's history is the earlier ordinary Saturdays. Sunday and every
                    other day are exactly as deployed. Nothing is added on top: the lift moves from ordinary Saturdays to payday Saturdays.
  stacked (SECONDARY)  the deployed correction stays as it is, and the payday Saturdays get an extra offset: n / (n + 10) x the mean residual
                    (actual minus corrected) of the payday Saturdays in earlier folds, none until 4 such days. K = 10 is fixed here before any run.

The scope is only the Saturday after a Friday payday, and it is not widened after the fact.

Checks on whether the result means anything:

  placebo     each rule re-run with the "payday" label given to 1,000 random sets of backtest Saturdays of the same size (the label is shuffled over ALL backtest
              Saturdays, real payday ones included). It shows how much a rule like this gains or loses when the label carries no information. The payday label was
              found by looking at the model's errors on these same days, so it will usually rank well: the placebo shows the scale of the noise, not that
              the selection is clean.
  new days    the three Saturdays after Friday paydays among the 56 days from 7 September 2026 (26 Sep, 24 Oct, 31 Oct), the only ones not used to find the
              pattern, treated as an 11th block by the same rule. Two were over-predicted by about 2,000 pounds, against the pattern. Three days confirm nothing,
              but they are the real test, because no rule could have been fitted to them.

Models: the post model and the hydro model, as in Series 54-64.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  look at     for each rule, the change in MAE against the deployed package over all 585 days with a 95% moving-block bootstrap interval; split into the payday
              Saturdays and the ordinary Saturdays (the split should help both: more lift where it was short, less where it overshot); in how many folds it helps;
              the placebo percentile; the new days
  expected    a gain of a few pounds a day at most: 22 payday Saturdays and 55 ordinary ones carry it out of 585 days
  would matter  split: an overall interval below zero AND a placebo percentile under 5% AND the new days not worse. The stacked rule is only compared with it
  limits      no pass/fail gate; nothing in production changes; adding it would be a new correction rule in the weekly log, decided by Kevin and labelled as a bias adjustment

Run:  python -m src.experiments.run_series65_payday_saturday_correction list | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF
from src.experiments.run_series61_payday_rules import add_payday_rule_columns
from src.experiments.series19_21_common import DATE_COL, TARGET_COL, load_protocol, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series65"
STACKED_K = 10.0
MIN_EVENTS = s24.MIN_HISTORY  # 4, the deployed correction's own minimum
PLACEBOS = 1000
NEW_FOLD = 10  # the 56 new days are scored as an 11th block


def extended_days() -> pd.DataFrame:
    """The 585 backtest days (folds 0-9) followed by the 56 new days as fold 10, with what the deployed correction needs."""
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    new = df[df[DATE_COL] >= CUTOFF]
    add = pd.DataFrame({"date": pd.to_datetime(new[DATE_COL]).to_numpy(), "actual": new[TARGET_COL].to_numpy(), "fold_id": NEW_FOLD})
    out = pd.concat([days[["date", "actual", "fold_id"]], add], ignore_index=True)
    out["weekday"] = out["date"].dt.day_name()
    return out


def payday_saturday_mask(ext: pd.DataFrame) -> np.ndarray:
    """Saturdays after a Friday that is a payday under the last working day, the 25th or the 28th (rolled back off weekends and bank holidays)."""
    flags = add_payday_rule_columns(pd.DataFrame({DATE_COL: ext["date"].to_numpy()}))
    return (flags["friday_payday_weekend"].to_numpy() == 1) & (ext["date"].dt.dayofweek.to_numpy() == 5)


def raw_predictions(ext: pd.DataFrame, mon_thu: str) -> np.ndarray:
    """The deployed package before any correction, for every day of `ext` (backtest then new days)."""
    bt = s54.load_bags("backtest", s54.RESULTS, {"none": []}, s54.RESULTS)
    nd = s54.load_bags("new_days", s54.RESULTS, {"none": []}, s54.RESULTS)
    dow = ext["date"].dt.dayofweek.to_numpy()
    in_bt = ext["fold_id"].to_numpy() < NEW_FOLD

    def pick(bags: dict, d: np.ndarray) -> np.ndarray:
        return np.select([d == 4, d == 5, d == 6], [bags[("Friday", "none")], bags[("Saturday", "none")], bags[("Sunday", "none")]],
                         default=bags[(mon_thu, "none")])

    return np.concatenate([pick(bt, dow[in_bt]), pick(nd, dow[~in_bt])])


def deployed_correction(ext: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    return s24.correct(ext, raw, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])


def split_correction(ext: pd.DataFrame, raw: np.ndarray, event: np.ndarray) -> np.ndarray:
    """The deployed correction, with Saturdays corrected as two groups: `event` days and the other Saturdays. Sunday is as deployed."""
    out = s24.correct(ext, raw, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=("Sunday",))  # Saturday left raw here
    fold, a = ext["fold_id"].to_numpy(), ext["actual"].to_numpy()
    dates = ext["date"].to_numpy()
    saturday = ext["date"].dt.dayofweek.to_numpy() == 5
    error = raw - a
    half_life, strength = CORRECTION["half_life_weeks"], CORRECTION["strength"]

    def bias(history: np.ndarray, start) -> float:
        age_weeks = (start - dates[history]) / np.timedelta64(7, "D")
        weights = 0.5 ** (age_weeks / half_life)
        return float(np.sum(weights * error[history]) / np.sum(weights))

    for k in sorted(set(fold.tolist()))[1:]:
        block = fold == k
        start = dates[block].min()
        for group, target in ((event, saturday & event), (~event, saturday & ~event)):
            history = (fold < k) & saturday & group
            if group is event and history.sum() < MIN_EVENTS:  # too few payday Saturdays yet: keep the all-Saturday bias, as deployed
                history = (fold < k) & saturday
            if history.sum() < MIN_EVENTS:
                continue
            out[block & target] = raw[block & target] - strength * bias(history, start)
    return out


def stacked_offset(ext: pd.DataFrame, corrected: np.ndarray, event: np.ndarray, shrink: float = STACKED_K) -> np.ndarray:
    """The deployed correction plus an extra offset on `event` days: n/(n+K) x the mean residual of earlier-fold events; none until MIN_EVENTS."""
    out = corrected.copy()
    fold, resid = ext["fold_id"].to_numpy(), ext["actual"].to_numpy() - corrected
    for k in sorted(set(fold.tolist()))[1:]:
        history = event & (fold < k)
        n = int(history.sum())
        if n >= MIN_EVENTS:
            out[event & (fold == k)] += n / (n + shrink) * float(resid[history].mean())
    return out


def summarise(label: str, ext: pd.DataFrame, a: np.ndarray, deployed: np.ndarray, rule_out: np.ndarray, event: np.ndarray, placebo_changes: np.ndarray) -> None:
    fold, dow = ext["fold_id"].to_numpy(), ext["date"].dt.dayofweek.to_numpy()
    in_bt = fold < NEW_FOLD
    diff = np.abs(rule_out - a) - np.abs(deployed - a)
    change, lo, hi = s23.masked_block_bootstrap(diff[in_bt], np.ones(int(in_bt.sum())))
    pay, other = event & in_bt & (fold >= 1), (dow == 5) & ~event & in_bt & (fold >= 1)
    folds_with = [k for k in range(1, NEW_FOLD) if (event & (fold == k)).any()]
    folds_better = sum(diff[event & (fold == k)].mean() < 0 for k in folds_with)
    print(f"  {label}")
    print(f"    all 585 days     change {change:+.2f}  [{lo:+.2f}, {hi:+.2f}]   (MAE {np.abs(deployed - a)[in_bt].mean():.2f} -> {np.abs(rule_out - a)[in_bt].mean():.2f})")
    print(f"    payday Saturdays {int(pay.sum())} (folds 1-9): mean change {diff[pay].mean():+.0f} per day; helped on {int((diff[pay] < 0).sum())}; "
          f"folds better {folds_better} of {len(folds_with)}")
    print(f"    other Saturdays  {int(other.sum())}: mean change {diff[other].mean():+.0f} per day; helped on {int((diff[other] < 0).sum())}")
    print(f"    placebo ({PLACEBOS} shuffled labels): median {np.median(placebo_changes):+.2f}, 5th to 95th percentile "
          f"{np.percentile(placebo_changes, 5):+.2f} to {np.percentile(placebo_changes, 95):+.2f}; the payday label sits at the "
          f"{100 * float((placebo_changes <= change).mean()):.0f}th percentile (share of shuffles at or below its change)")
    new_ev = event & ~in_bt
    new_other = (dow == 5) & ~event & ~in_bt
    print(f"    new days: payday Saturdays {[(round(float(a[i] - deployed[i])), round(float(a[i] - rule_out[i]))) for i in np.where(new_ev)[0]]} "
          f"(error as deployed, error with the rule); mean change {diff[new_ev].mean():+.0f} on them, {diff[new_other].mean():+.0f} on the {int(new_other.sum())} other new Saturdays, "
          f"{diff[~in_bt].mean():+.2f} over all {int((~in_bt).sum())} new days")


def report() -> None:
    ext = extended_days()
    fold, a = ext["fold_id"].to_numpy(), ext["actual"].to_numpy()
    in_bt = fold < NEW_FOLD
    event = payday_saturday_mask(ext)
    saturday = ext["date"].dt.dayofweek.to_numpy() == 5
    n_bt = int((event & in_bt).sum())
    print(f"{int(in_bt.sum())} backtest days, {len(ext) - int(in_bt.sum())} new days; payday Saturdays: {n_bt} in the backtest, {int((event & ~in_bt).sum())} among the new days")
    rng = np.random.default_rng(65)
    pool = np.where(saturday & in_bt)[0]
    shuffles = []
    for _ in range(PLACEBOS):
        mask = np.zeros(len(ext), bool)
        mask[rng.choice(pool, size=n_bt, replace=False)] = True
        mask |= event & ~in_bt  # the new days keep their real label; the placebo is scored on the backtest only
        shuffles.append(mask)

    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        raw = raw_predictions(ext, mon_thu)
        deployed = deployed_correction(ext, raw)
        resid = a - deployed
        print(f"\n{name.upper()}  residual after the deployed correction (actual minus corrected; + = too low): payday Saturdays {resid[event & in_bt].mean():+.0f} "
              f"(median {np.median(resid[event & in_bt]):+.0f}), other Saturdays {resid[saturday & ~event & in_bt].mean():+.0f}")
        split = split_correction(ext, raw, event)
        split_placebo = np.array([float((np.abs(split_correction(ext, raw, m) - a) - np.abs(deployed - a))[in_bt].mean()) for m in shuffles])
        summarise("split (PRIMARY)", ext, a, deployed, split, event, split_placebo)
        stacked = stacked_offset(ext, deployed, event)
        stacked_placebo = np.array([float((np.abs(stacked_offset(ext, deployed, m) - a) - np.abs(deployed - a))[in_bt].mean()) for m in shuffles])
        summarise(f"stacked, K={STACKED_K:.0f} (SECONDARY)", ext, a, deployed, stacked, event, stacked_placebo)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str) -> None:
    if step == "list":
        ext = extended_days()
        event = payday_saturday_mask(ext)
        in_bt = ext["fold_id"].to_numpy() < NEW_FOLD
        assert (ext.loc[event, "date"].dt.dayofweek == 5).all()
        print(f"{len(ext)} days ({int(in_bt.sum())} backtest, {int((~in_bt).sum())} new); payday Saturdays {int((event & in_bt).sum())} + {int((event & ~in_bt).sum())}")
        print("payday Saturdays by fold:", pd.Series(event.astype(int)).groupby(ext["fold_id"].to_numpy()).sum().to_dict())
        print("new-day payday Saturdays:", [str(d.date()) for d in ext.loc[event & ~in_bt, "date"]])
        raw = raw_predictions(ext, "Monday-Thursday")
        deployed = deployed_correction(ext, raw)
        a = ext["actual"].to_numpy()
        mae = float(np.abs(deployed - a)[in_bt].mean())
        print(f"post model after the deployed correction, 585 days: MAE {mae:.2f} (Series 61 printed 865.36 as deployed)")
        assert abs(mae - 865.36) < 0.01, "the saved predictions do not reproduce the deployed package"
        # with no payday label the split must reproduce the deployed correction exactly, and the stacked rule must change nothing
        none = np.zeros(len(ext), bool)
        assert np.allclose(split_correction(ext, raw, none), deployed), "the split with an empty label differs from the deployed correction"
        assert np.allclose(stacked_offset(ext, deployed, none), deployed)
        split = split_correction(ext, raw, event)
        changed = ~np.isclose(split, deployed)
        print(f"split with the real label changes {int(changed.sum())} days, all Saturdays: {bool((ext.loc[changed, 'date'].dt.dayofweek == 5).all())}; "
              f"the first fold it changes the payday group: "
              f"{int(ext.loc[changed & event, 'fold_id'].min())}, the ordinary group: {int(ext.loc[changed & ~event, 'fold_id'].min())}")
        print(f"stacked changes {int((~np.isclose(stacked_offset(ext, deployed, event), deployed)).sum())} days")
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "report"])
    args = parser.parse_args()
    main(args.step)
