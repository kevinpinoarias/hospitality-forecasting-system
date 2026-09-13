"""
Feature-set definitions and permutation generation for the feature
engineering sweep (src/experiments/run_feature_sweep.py).

Design of experiments
----------------------
A true exhaustive search over all 2^23 subsets of the candidate pool is
computationally infeasible and, more importantly, uninterpretable - a
"best" 2^23-th subset tells you nothing about *why* it's best. Instead
this uses a structured, hypothesis-driven design:

1. BASELINE - the current production feature set (control).
2. LEAVE-ONE-OUT - remove each of the 15 baseline features individually.
   Tests whether every feature currently in the model is actually earning
   its place, not just assumed to.
3. ADD-ONE - add each of the 23 candidate features individually to the
   baseline. Tests each candidate's own marginal contribution in
   isolation.
4. ADD-GROUP - add each thematic group of candidates (see
   CANDIDATE_FEATURE_GROUPS) to the baseline as a whole. Tests whether
   related features have a combined/synergistic effect beyond what any
   one of them shows alone.
5. ADD-ALL - every candidate added at once. An upper-bound "kitchen sink"
   test, and a check for the multicollinearity/overfitting risk of just
   throwing everything in.
6. ADD-TOP-K COMBINATIONS - built in a second phase (see
   run_feature_sweep.py) from whichever individual candidates ranked best
   in step 3, once those results exist - tests whether the best
   individual performers also combine well, or are partially redundant
   with each other.

Every feature set is registered with a name and a plain-language
hypothesis - this is the record of "what was tried and why", not just
"what scored what".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.experiments.adapters import FEATURES as BASELINE_FEATURES

# The 23 already-engineered-but-currently-unused columns in
# data/features/engineered_features.csv, grouped by theme. Excludes
# forecast_error itself (leakage - directly encodes the target) and the
# date-typed intermediate columns (payday, prev_payday, next_payday) that
# exist only to derive other features, not to be inputs themselves.
CANDIDATE_FEATURE_GROUPS: dict[str, list[str]] = {
    "sales_error_history": [
        "lag_7_sales", "rolling_7_sales", "rolling_14_sales",
        "lag_1_fe", "lag_7_fe", "rolling_7_fe",
    ],
    "calendar_extra": ["year", "is_weekend", "day_of_week_cos"],
    "payday_extra": ["is_payday", "days_to_payday"],
    "bank_holiday_extra": ["days_since_bank_holiday"],
    "school_holiday": [
        "is_christmas_break", "is_summer_break", "is_easter_break", "is_school_holiday",
    ],
    "weather_raw": ["max_temp", "rain_mm", "sun_hours"],
    "weather_derived_extra": [
        "is_hot_for_scotland", "is_dry_day", "temp_anomaly_14d", "warm_streak_len",
    ],
}

ALL_CANDIDATES: list[str] = [f for group in CANDIDATE_FEATURE_GROUPS.values() for f in group]

LEAKAGE_EXCLUDED = ["forecast_error"]
NON_FEATURE_INTERMEDIATES = ["payday", "prev_payday", "next_payday"]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: tuple[str, ...]
    category: str
    hypothesis: str
    added: tuple[str, ...] = field(default_factory=tuple)
    removed: tuple[str, ...] = field(default_factory=tuple)


def build_baseline_and_screening_sets() -> list[FeatureSet]:
    """Phase 1: baseline, leave-one-out, add-one, add-group, add-all."""
    sets: list[FeatureSet] = []

    sets.append(FeatureSet(
        name="baseline_15",
        features=tuple(BASELINE_FEATURES),
        category="baseline",
        hypothesis="Control: the current production feature set, unchanged.",
    ))

    for feat in BASELINE_FEATURES:
        remaining = tuple(f for f in BASELINE_FEATURES if f != feat)
        sets.append(FeatureSet(
            name=f"leave_out__{feat}",
            features=remaining,
            category="leave_one_out",
            hypothesis=f"Does removing '{feat}' from the baseline hurt accuracy? "
                       f"If not, it may not be earning its place in the model.",
            removed=(feat,),
        ))

    for feat in ALL_CANDIDATES:
        sets.append(FeatureSet(
            name=f"add_one__{feat}",
            features=tuple(BASELINE_FEATURES) + (feat,),
            category="add_one",
            hypothesis=f"Does adding '{feat}' alone improve on the baseline?",
            added=(feat,),
        ))

    for group_name, group_features in CANDIDATE_FEATURE_GROUPS.items():
        sets.append(FeatureSet(
            name=f"add_group__{group_name}",
            features=tuple(BASELINE_FEATURES) + tuple(group_features),
            category="add_group",
            hypothesis=f"Does the '{group_name}' group ({', '.join(group_features)}) improve on the "
                       f"baseline as a whole, beyond what any single member of it shows alone?",
            added=tuple(group_features),
        ))

    sets.append(FeatureSet(
        name="add_all_candidates",
        features=tuple(BASELINE_FEATURES) + tuple(ALL_CANDIDATES),
        category="add_all",
        hypothesis="Upper-bound test: every designed-but-unused feature added at once. "
                   "Tests both best-case ceiling and multicollinearity/overfitting risk.",
        added=tuple(ALL_CANDIDATES),
    ))

    return sets


def build_top_k_combination_sets(ranked_candidates: list[str], k_values: tuple[int, ...] = (3, 5, 8)) -> list[FeatureSet]:
    """
    Phase 2: combine the top-K individually-best-performing candidates
    (ranked_candidates, best first, from the add-one phase's results).
    Tests whether the strongest individual performers combine well or are
    partially redundant with one another (e.g. two weather features
    carrying overlapping signal).
    """
    sets = []
    for k in k_values:
        top_k = ranked_candidates[:k]
        if not top_k:
            continue
        sets.append(FeatureSet(
            name=f"add_top{k}_individual",
            features=tuple(BASELINE_FEATURES) + tuple(top_k),
            category="add_combo",
            hypothesis=f"Do the top {k} individually-best candidates ({', '.join(top_k)}) "
                       f"combine additively, or show diminishing/negative returns from redundancy?",
            added=tuple(top_k),
        ))
    return sets
