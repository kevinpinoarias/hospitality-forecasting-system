"""
When each forecast of the log is issued, which days it covers and when it is late (section 2 of the log's PREREGISTRATION.md).

  NW  next week   issued every Monday 08:00   covers the following Monday to Sunday   deadline: Monday 23:59
  TW  this week   issued every Monday 08:00   covers the current Monday to Sunday     deadline: Monday 10:00 (before the venue opens)
  TD  tomorrow    issued every evening 18:00  covers the next day                     deadline: 23:59 on the day it is issued

A run issued late (a catch-up after the computer was off) still forecasts, but only days that have not started: a day is forecast only if it is after the issue date, or the issue date
itself and the issue time is before the venue opens. Anything issued after its deadline is labelled `late`.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

NEXT_WEEK, THIS_WEEK, TOMORROW = "NW", "TW", "TD"
FORECAST_TYPES = (NEXT_WEEK, THIS_WEEK, TOMORROW)
OPENING_HOUR = 10
TRIGGERS = {"monday": (NEXT_WEEK, THIS_WEEK), "evening": (TOMORROW,)}


def week_monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def allowed(target: date, issue: datetime) -> bool:
    """A day can be forecast only if it has not started: after the issue date, or the issue date itself before the venue opens."""
    return target > issue.date() or (target == issue.date() and issue.hour < OPENING_HOUR)


def target_dates(forecast_type: str, issue: datetime) -> list[date]:
    monday = week_monday(issue.date())
    if forecast_type == NEXT_WEEK:
        days = [monday + timedelta(days=7 + i) for i in range(7)]
    elif forecast_type == THIS_WEEK:
        days = [monday + timedelta(days=i) for i in range(7)]
    elif forecast_type == TOMORROW:
        days = [issue.date() + timedelta(days=1)]
    else:
        raise ValueError(f"unknown forecast type {forecast_type!r}")
    return [d for d in days if allowed(d, issue)]


def deadline(forecast_type: str, issue: datetime) -> datetime:
    """The time by which a forecast issued in the week/day of `issue` must be made (same time zone as `issue`)."""
    monday = week_monday(issue.date())
    tz = issue.tzinfo
    if forecast_type == NEXT_WEEK:
        return datetime.combine(monday, time(23, 59, 59), tzinfo=tz)
    if forecast_type == THIS_WEEK:
        return datetime.combine(monday, time(OPENING_HOUR, 0), tzinfo=tz)
    if forecast_type == TOMORROW:
        return datetime.combine(issue.date(), time(23, 59, 59), tzinfo=tz)
    raise ValueError(f"unknown forecast type {forecast_type!r}")


def on_time(forecast_type: str, issue: datetime) -> bool:
    return issue <= deadline(forecast_type, issue)


def forecast_types_for(trigger: str) -> tuple[str, ...]:
    """The forecast types a trigger name stands for ('monday', 'evening', or one type code)."""
    if trigger in TRIGGERS:
        return TRIGGERS[trigger]
    if trigger in FORECAST_TYPES:
        return (trigger,)
    raise ValueError(f"unknown trigger {trigger!r}")
