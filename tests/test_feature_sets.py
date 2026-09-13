"""
Tests for src/experiments/feature_sets.py - the design-of-experiments
registry for the feature engineering sweep. These guard the two
properties that matter most for a "rigorous" experiment log: every
generated set is uniquely named and traceable, and the known data-leakage
column can never appear in any candidate set.
"""

from __future__ import annotations

from src.experiments.adapters import FEATURES as BASELINE_FEATURES
from src.experiments.feature_sets import (
    ALL_CANDIDATES,
    CANDIDATE_FEATURE_GROUPS,
    LEAKAGE_EXCLUDED,
    build_baseline_and_screening_sets,
    build_top_k_combination_sets,
)


def test_candidate_pool_has_23_features():
    assert len(ALL_CANDIDATES) == 23


def test_candidate_pool_has_no_duplicates_across_groups():
    assert len(ALL_CANDIDATES) == len(set(ALL_CANDIDATES))


def test_leakage_column_never_appears_in_any_group():
    for group_features in CANDIDATE_FEATURE_GROUPS.values():
        for leaked in LEAKAGE_EXCLUDED:
            assert leaked not in group_features


def test_baseline_features_never_appear_as_candidates():
    """A candidate that's already in the baseline shouldn't also be
    offered as something to 'add' - that would be a no-op, not a real
    experiment."""
    for feat in ALL_CANDIDATES:
        assert feat not in BASELINE_FEATURES


def test_screening_sets_have_expected_counts():
    sets = build_baseline_and_screening_sets()
    by_category = {}
    for s in sets:
        by_category.setdefault(s.category, []).append(s)

    assert len(by_category["baseline"]) == 1
    assert len(by_category["leave_one_out"]) == len(BASELINE_FEATURES)
    assert len(by_category["add_one"]) == len(ALL_CANDIDATES)
    assert len(by_category["add_group"]) == len(CANDIDATE_FEATURE_GROUPS)
    assert len(by_category["add_all"]) == 1


def test_all_set_names_are_unique():
    sets = build_baseline_and_screening_sets()
    names = [s.name for s in sets]
    assert len(names) == len(set(names))


def test_leave_one_out_produces_14_features():
    sets = build_baseline_and_screening_sets()
    loo = [s for s in sets if s.category == "leave_one_out"]
    for s in loo:
        assert len(s.features) == len(BASELINE_FEATURES) - 1
        assert set(s.features) < set(BASELINE_FEATURES)


def test_add_one_produces_16_features_including_baseline():
    sets = build_baseline_and_screening_sets()
    add_one = [s for s in sets if s.category == "add_one"]
    for s in add_one:
        assert len(s.features) == len(BASELINE_FEATURES) + 1
        assert set(BASELINE_FEATURES) < set(s.features)


def test_add_all_includes_every_candidate_and_baseline_feature():
    sets = build_baseline_and_screening_sets()
    add_all = next(s for s in sets if s.name == "add_all_candidates")
    assert set(add_all.features) == set(BASELINE_FEATURES) | set(ALL_CANDIDATES)


def test_no_feature_set_contains_the_leaked_column():
    sets = build_baseline_and_screening_sets()
    for s in sets:
        for leaked in LEAKAGE_EXCLUDED:
            assert leaked not in s.features, f"{s.name} contains leaked column {leaked}"


def test_top_k_combination_sets_use_ranked_order():
    ranked = ["max_temp", "is_payday", "rain_mm", "year", "is_weekend", "warm_streak_len", "is_dry_day", "day_of_week_cos"]
    sets = build_top_k_combination_sets(ranked, k_values=(3, 5))
    top3 = next(s for s in sets if s.name == "add_top3_individual")
    top5 = next(s for s in sets if s.name == "add_top5_individual")
    assert set(top3.added) == set(ranked[:3])
    assert set(top5.added) == set(ranked[:5])
    assert set(top3.features) == set(BASELINE_FEATURES) | set(ranked[:3])
