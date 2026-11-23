"""
Tests for the exploratory event calendars (src/features/events.py, config/events/). They guard the shape of the lists and the
definitions of the flags, not the sources' completeness (which docs/EVENTS_EXPLORATION_2026_10.md states).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import events


@pytest.fixture(scope="module")
def football() -> pd.DataFrame:
    return events.football_calendar()


@pytest.fixture(scope="module")
def concerts() -> pd.DataFrame:
    return events.concert_calendar()


def test_the_lists_have_the_columns_the_loaders_expect():
    matches = pd.read_csv(events.EVENTS_DIR / "football_matches.csv", parse_dates=["date"])
    shows = pd.read_csv(events.EVENTS_DIR / "concert_days.csv", parse_dates=["date"])
    assert list(matches.columns) == ["team", "date", "opponent", "venue"]
    assert set(matches["team"]) == {"Celtic", "Rangers"} and set(matches["venue"]) <= {"H", "A", "N"}
    assert list(shows.columns) == ["venue", "date", "capacity"]
    assert set(shows["venue"]) == set(events.VENUES)
    assert not shows.duplicated(["venue", "date"]).any()  # one row per venue per day


def test_there_is_one_row_per_day_and_no_gaps(football, concerts):
    for frame in (football, concerts):
        assert frame["date"].is_unique and frame["date"].is_monotonic_increasing
        assert (frame["date"].diff().dropna() == pd.Timedelta(days=1)).all()


def test_a_known_old_firm_day_is_a_derby_and_a_home_game_for_the_host(football):
    day = football[football["date"] == "2026-09-20"].iloc[0]  # Celtic at home to Rangers (both lists agree)
    assert day["old_firm"] == 1 and day["celtic_home"] == 1 and day["rangers_away"] == 1 and day["rangers_home"] == 0


def test_the_saturday_flag_is_only_ever_set_on_a_saturday_with_a_home_game(football):
    flagged = football[football["sat_big_home"] == 1]
    assert len(flagged) > 0
    assert (flagged["date"].dt.dayofweek == 5).all()
    assert ((flagged["celtic_home"] + flagged["rangers_home"]) > 0).all()
    home_saturdays = football[(football["any_home_big"] == 1) & (football["date"].dt.dayofweek == 5)]
    assert len(home_saturdays) == len(flagged)


def test_neutral_ground_games_are_not_home_games(football):
    # the 2025 League Cup semi-final, Celtic v Rangers at Hampden, is a derby but neither side's home game
    day = football[football["date"] == "2025-11-02"].iloc[0]
    assert day["old_firm"] == 1 and day["celtic_home"] == 0 and day["rangers_home"] == 0 and day["any_home_big"] == 0


def test_two_shows_on_one_night_add_their_capacities_once_each(concerts):
    both = concerts[(concerts["hydro_show"] == 1) & (concerts["o2_show"] == 1)]
    assert len(both) > 0
    cap = {"hydro_show": 14.3, "o2_show": 2.55, "armadillo_show": 3.0}
    for _, row in both.iterrows():
        expected = sum(cap[c] for c in cap if row[c] == 1)
        assert row["concert_capacity_k"] == pytest.approx(expected)


def test_a_known_hydro_night_is_flagged(concerts):
    day = concerts[concerts["date"] == "2026-09-30"].iloc[0]  # Westlife at the Hydro
    assert day["hydro_show"] == 1 and day["any_show"] == 1


def test_days_without_any_show_have_zero_capacity(concerts):
    quiet = concerts[concerts["any_show"] == 0]
    assert len(quiet) > 0 and (quiet["concert_capacity_k"] == 0).all()


def test_scotland_flags_split_home_away_neutral_and_tournament():
    sc = events.scotland_calendar().set_index("date")
    assert sc.loc["2026-09-29", "scotland_home"] == 1 and sc.loc["2026-09-29", "scotland_away"] == 0         # Switzerland at Hampden
    assert sc.loc["2025-09-08", "scotland_away"] == 1 and sc.loc["2025-09-08", "scotland_home"] == 0         # Belarus, neutral venue in Hungary
    assert sc.loc["2026-06-14", "scotland_tournament"] == 1 and sc.loc["2026-06-14", "scotland_away"] == 0   # World Cup opener, 2am UK
    assert sc.loc["2024-06-14", "scotland_tournament"] == 1                                                  # Euro 2024 opener


def test_a_scotland_game_sets_exactly_one_kind_and_the_plays_flag():
    sc = events.scotland_calendar()
    kinds = sc[["scotland_home", "scotland_away", "scotland_tournament"]].sum(axis=1)
    assert kinds.max() == 1
    assert ((kinds > 0).astype(int) == sc["scotland_plays"]).all()
    assert sc["scotland_plays"].sum() == 34  # 36 listed matches, the two scheduled for November 2026 are beyond the calendar's end


def test_the_event_calendar_carries_every_flag_once_per_day():
    cal = events.event_calendar()
    assert cal["date"].is_unique and len(cal) == len(events.football_calendar())
    assert {"scotland_home", "sat_big_home", "hydro_show"} <= set(cal.columns)
