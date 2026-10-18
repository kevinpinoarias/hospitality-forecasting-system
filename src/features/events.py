"""
Event calendars as daily flags: Celtic / Rangers football and concerts at the three Glasgow music venues.

EXPLORATORY (Series 36). Nothing here is wired into the feature pipeline or the final model; these are the inputs to the
checks in src/experiments/run_series36_event_checks.py. The curated lists live in config/events/:

  football_matches.csv   team, date, opponent, venue (H home, A away, N neutral) - Celtic and Rangers, all competitions
  concert_days.csv       venue, date, capacity - one row per venue per day with a show (OVO Hydro, O2 Academy, SEC Armadillo)

Sources and their limits are in docs/EVENTS_EXPLORATION_2026_10.md (the football list is incomplete in places; the concert
list misses non-music events and roughly a tenth of the Hydro's shows). The lists are facts about public events, not venue data.
"""

from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT

EVENTS_DIR = PROJECT_ROOT / "config" / "events"
FIRST_DAY, LAST_DAY = "2024-01-01", "2026-10-31"
SATURDAY = 5
VENUES = {"OVO Hydro": "hydro_show", "O2 Academy": "o2_show", "SEC Armadillo": "armadillo_show"}


def football_calendar(first: str = FIRST_DAY, last: str = LAST_DAY) -> pd.DataFrame:
    """One row per day: whether Celtic / Rangers play, at home or away, a derby, and the Saturday home-match flag."""
    matches = pd.read_csv(EVENTS_DIR / "football_matches.csv", parse_dates=["date"])
    days = pd.DataFrame({"date": pd.date_range(first, last)})

    def has(team: str, venue: str | None = None) -> pd.Series:
        m = matches[matches["team"] == team]
        if venue is not None:
            m = m[m["venue"] == venue]
        return days["date"].isin(m["date"]).astype(int)

    days["celtic_plays"], days["rangers_plays"] = has("Celtic"), has("Rangers")
    days["celtic_home"], days["celtic_away"] = has("Celtic", "H"), has("Celtic", "A")
    days["rangers_home"], days["rangers_away"] = has("Rangers", "H"), has("Rangers", "A")
    derby = set(matches[(matches["team"] == "Celtic") & (matches["opponent"] == "Rangers")]["date"]) | \
        set(matches[(matches["team"] == "Rangers") & (matches["opponent"] == "Celtic")]["date"])
    days["old_firm"] = days["date"].isin(list(derby)).astype(int)
    home = (days["celtic_home"] + days["rangers_home"]) > 0
    days["any_home_big"] = home.astype(int)
    days["sat_big_home"] = (home & (days["date"].dt.dayofweek == SATURDAY)).astype(int)
    return days


def concert_calendar(first: str = FIRST_DAY, last: str = LAST_DAY) -> pd.DataFrame:
    """One row per day: a show at each venue, whether there is any, and the summed capacity (each venue counted once a day)."""
    shows = pd.read_csv(EVENTS_DIR / "concert_days.csv", parse_dates=["date"]).drop_duplicates(["venue", "date"])
    days = pd.DataFrame({"date": pd.date_range(first, last)})
    for venue, col in VENUES.items():
        days[col] = days["date"].isin(shows[shows["venue"] == venue]["date"]).astype(int)
    days["any_show"] = (days[list(VENUES.values())].sum(axis=1) > 0).astype(int)
    cap = shows.groupby("date")["capacity"].sum()
    days["concert_capacity_k"] = days["date"].map(cap).fillna(0.0) / 1000.0
    return days


def event_calendar(first: str = FIRST_DAY, last: str = LAST_DAY) -> pd.DataFrame:
    """Football and concert flags side by side, one row per day."""
    return football_calendar(first, last).merge(concert_calendar(first, last), on="date")
