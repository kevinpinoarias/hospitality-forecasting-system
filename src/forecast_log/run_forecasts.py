"""
One run of the forecast log: make the forecasts that are due, append them to the log, commit and push (rules: PREREGISTRATION.md in the `forecasting-log` repository).

  python -m src.forecast_log.run_forecasts --trigger monday      next week (NW) and this week (TW)               scheduled Monday 12:30 (scripts/run_forecast_log.cmd weekly)
  python -m src.forecast_log.run_forecasts --trigger evening     tomorrow (TD) only                              by hand
  add --dry-run to compute and print everything without writing, committing or pushing
  python -m src.forecast_log.run_forecasts --trigger monday --reissue-of 20261116T1230-monday   a withdrawn run made again on newer sales data (see below)

For every forecast type the run makes the three models (portfolio, post, hydro) without the manager's forecast (it only reaches us after the day; the stand-in takes its place), with the
weather the live API would use, and logs the seasonal baseline as a benchmark; the manager's own number is added afterwards, when the export brings it (score_forecasts.py). The weather the run
used is saved to log/weather_used.csv. A run issued after its deadline is labelled `late`. A run that cannot proceed (models changed, features missing, ...) writes an `aborted` line to runs.csv
with the reason and makes no forecast. Settings: config/forecast_log.yaml (the log repository and the model version to use), overridden by config/forecast_log.local.yaml (not tracked).
`also_run: [v1.11, v1.12, v1.13, v1.14]` (a setting, off by default) issues further registered versions' hydro arms beside the main version, as rows named `hydro_v1.11` and so on; each version must be registered
and its models and correction history present, or the run aborts and says so. v1.13 and v1.14 carry specialist pieces for 26 December to 8 January (predict.with_year_end_specialists).
`holiday_rule: true` (a setting, off by default) adds the holiday rule's rows (model `holiday_rule`, src/forecast_log/holiday_rule.py: 0 on 25 December and 1 January, the earlier Boxing Days' mean on 26 December)
for those days only, beside the models' own rows.
`--reissue-of` makes a withdrawn run again (one issued before that week's export arrived) on the sales data now available: the same days and leads as that run, the weather that run saved (what was
known that Monday, nothing newer), and the real issue time, so a reissue after the deadline is labelled `late`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.experiments import run_series35_holdout as h
from src.forecast_log import holiday_rule as hrule
from src.forecast_log import inputs, logio, predict, schedule, v1
from src.models.final_model import DATE_COL, TARGET_COL

FEATURES = PROJECT_ROOT / "data" / "features" / "engineered_features.csv"
LOCK = PROJECT_ROOT / ".forecast_log.lock"
STALE_LOCK_SECONDS = 2 * 3600


def load_settings() -> dict:
    """The settings, with log_repo as an absolute path: a relative one is read from the project folder (not from wherever the run was started)."""
    settings = {"log_repo": "../forecasting-log", "model_version": v1.VERSION}
    for name in ("forecast_log.yaml", "forecast_log.local.yaml"):
        path = PROJECT_ROOT / "config" / name
        if path.exists():
            settings.update({k: v for k, v in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items() if v is not None})
    log_repo = Path(str(settings["log_repo"]))
    settings["log_repo"] = str(log_repo if log_repo.is_absolute() else (PROJECT_ROOT / log_repo).resolve())
    return settings


class RunLock:
    """One run at a time: a lock file that is ignored once it is older than two hours (a crashed run)."""

    def __enter__(self):
        if LOCK.exists() and time.time() - LOCK.stat().st_mtime < STALE_LOCK_SECONDS:
            raise RuntimeError("another forecast-log run is in progress (lock file present)")
        LOCK.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        LOCK.unlink(missing_ok=True)


def input_hash(features_sha: str, weather: pd.DataFrame) -> str:
    """The run's inputs in one hash: the engineered-features file and the weather laid over it."""
    return hashlib.sha256((features_sha + "|" + weather.to_csv(index=False)).encode()).hexdigest()[:32]


def weather_rows(run_id: str, weather: pd.DataFrame, fetched: str | None = None) -> list[dict]:
    fetched = fetched or logio.stamp()
    return [{"run_id": run_id, "date": d.strftime("%Y-%m-%d"), "max_temp": t, "rain_mm": r, "sun_hours": s, "fetched_at": fetched}
            for d, t, r, s in zip(weather[DATE_COL], weather["max_temp"], weather["rain_mm"], weather["sun_hours"])]


def label(forecast_type: str, issue: datetime, made: datetime) -> str:
    """`on_time` if the forecast was really made by the deadline of the week it is issued for (`issue`), else `late`."""
    return "on_time" if made <= schedule.deadline(forecast_type, issue) else "late"


def withdrawn_run(log_repo: Path, run_id: str) -> tuple[datetime, pd.DataFrame, str]:
    """The issue time, saved weather and weather fetch time of a withdrawn run, so it can be made again on newer sales data with what was known that day."""
    runs = pd.read_csv(log_repo / "log" / "runs.csv", dtype=str)
    row = runs[runs["run_id"] == run_id]
    if row.empty or row.iloc[0]["status"] != "withdrawn":
        raise RuntimeError(f"run {run_id} is not a withdrawn run in runs.csv: only a withdrawn run is reissued")
    saved = pd.read_csv(log_repo / "log" / "weather_used.csv")
    saved = saved[saved["run_id"] == run_id]
    weather = saved[["date", "max_temp", "rain_mm", "sun_hours"]].rename(columns={"date": DATE_COL}).reset_index(drop=True)
    weather[DATE_COL] = pd.to_datetime(weather[DATE_COL]).dt.normalize()
    return datetime.fromisoformat(row.iloc[0]["issue_time"]), weather, (str(saved["fetched_at"].iloc[0]) if len(saved) else "")


def build_rows(issue: datetime, trigger: str, run_id: str, log_repo: Path, version: str, commit: str,
               also_run: tuple[str, ...] = (), with_holiday_rule: bool = False, made: datetime | None = None,
               saved_weather: tuple[pd.DataFrame, str] | None = None) -> tuple[list[dict], list[str], pd.Timestamp, list[dict]]:
    """The log rows this run would write, notes about what was skipped, the date the sales data runs through, and the weather the run used.
    `also_run` lists further registered versions (v1.11) whose hydro arm is issued beside the main version's three models, as rows of their own.
    `with_holiday_rule` adds the holiday rule's row (model `holiday_rule`) on 25 December, 26 December and 1 January.
    For a reissue, `issue` is the withdrawn run's issue time (the days and leads count from it), `made` the real time and `saved_weather` that run's weather and fetch time."""
    made = made or issue
    full = h.load_engineered(FEATURES)
    known = full.dropna(subset=[TARGET_COL])
    last_known = known[DATE_COL].max()
    models = predict.load_models(log_repo, version)
    history = v1.load_correction_history(v1.correction_history_path(version))
    extras = [(m, v1.load_correction_history(v1.correction_history_path(m.version))) for m in (predict.load_models(log_repo, v) for v in also_run)]
    features_sha = h.file_sha256(FEATURES)
    if saved_weather is None:
        weather, errored = predict.fetch_run_weather(last_known, issue.date())
    else:  # what was known that day, for the days after the sales data now ends
        weather = saved_weather[0][saved_weather[0][DATE_COL] > last_known].reset_index(drop=True)
        errored = weather.empty
    through = inputs.hydro_list_through()
    rows: list[dict] = []
    notes: list[str] = ["weather service unavailable: historical averages used"] if errored else []
    for ftype in schedule.forecast_types_for(trigger):
        targets = [d for d in schedule.target_dates(ftype, issue) if pd.Timestamp(d) > last_known]  # never forecast a day with a known actual
        if not targets:
            notes.append(f"{ftype}: no day to forecast (all started or already known)")
            continue
        unknown = inputs.unknown_hydro_thursdays(targets, through)
        if unknown:
            notes.append(f"{ftype}: hydro model skipped on {sorted(str(d) for d in unknown)} (the Hydro list reaches {through})")
        base = {"run_id": run_id, "issue_time": made.isoformat(timespec="seconds"), "forecast_type": ftype, "model_version": models.version, "data_through": str(last_known.date()),
                "hydro_list_through": str(through), "code_commit": commit, "on_time": label(ftype, issue, made), "input_hash": input_hash(features_sha, weather), "manager_arrived_at": ""}
        result = predict.forecast_days(models, full, last_known, [pd.Timestamp(d) for d in targets], history, weather, extras)
        for i, d in enumerate(targets):
            common = {**base, "target_date": str(d), "lead_days": (d - issue.date()).days, "variant": "no_manager", "weather_source": predict.weather_label(d, weather, errored, issue.date()),
                      "manager_forecast_used": round(float(result["forecast_sales_input"].iloc[i]), 2)}
            for model in predict.MODELS:
                if model == "hydro" and d in unknown:
                    continue
                fingerprint = models.portfolio_fingerprint if model == "portfolio" else models.combined_fingerprint
                rows.append({**common, "model": model, "forecast": round(float(result[model].iloc[i]), 2), "model_fingerprint": fingerprint})
            for extra, _ in extras:  # a further version's hydro arm: its own model name and version, and no forecast on a Hydro Thursday the list does not cover
                if d in unknown:
                    continue
                arm = predict.extra_arm_name(extra.version)
                rows.append({**common, "model": arm, "model_version": extra.version, "forecast": round(float(result[arm].iloc[i]), 2), "model_fingerprint": extra.combined_fingerprint,
                             "manager_forecast_used": round(float(result[arm + predict.STANDIN_SUFFIX].iloc[i]), 2)})  # the stand-in this version was given (the hybrid's differs from the main arm's)
            if with_holiday_rule:  # the rule's number on the closed days and Boxing Day only; the models' rows above stay what the models said
                value = hrule.rule_forecast(pd.Timestamp(d), hrule.boxing_day_level(known, pd.Timestamp(d), DATE_COL, TARGET_COL))
                if value is not None:
                    rows.append({**common, "model": hrule.RULE_MODEL, "model_version": hrule.RULE_VERSION, "forecast": round(value, 2), "model_fingerprint": hrule.fingerprint()})
        # the benchmark that needs no model, once per forecast type and day (the manager's number is added by the scoring once the export brings it)
        seasonal = predict.seasonal_baseline(full, last_known, [pd.Timestamp(d) for d in targets])
        for i, d in enumerate(targets):
            rows.append({**base, "target_date": str(d), "lead_days": (d - issue.date()).days, "variant": "", "weather_source": "", "model": "seasonal",
                         "forecast": round(float(seasonal[i]), 2), "manager_forecast_used": "", "model_fingerprint": ""})
    return rows, notes, last_known, weather_rows(run_id, weather, saved_weather[1] if saved_weather else None)


def run(trigger: str, now: datetime, dry_run: bool, log_repo: Path, version: str, also_run: tuple[str, ...] = (), with_holiday_rule: bool = False,
        reissue_of: str | None = None) -> int:
    run_id = f"{now:%Y%m%dT%H%M}-{trigger}"
    commit = v1.code_commit()
    types = "+".join(schedule.forecast_types_for(trigger))
    run_row = {"run_id": run_id, "issue_time": now.isoformat(timespec="seconds"), "trigger": trigger, "status": "ok", "reason": "", "data_through": "", "forecast_types_made": types,
               "code_commit": commit}
    rows: list[dict] = []
    weather: list[dict] = []
    try:
        if not dry_run:
            pending = logio.push_pending(log_repo)
            if pending == "push_failed":
                print("[run] warning: an earlier commit could not be pushed; will retry after this run")
        if reissue_of:
            issue, weather_saved, fetched = withdrawn_run(log_repo, reissue_of)
            rows, notes, last_known, weather = build_rows(issue, trigger, run_id, log_repo, version, commit, also_run, with_holiday_rule, made=now, saved_weather=(weather_saved, fetched))
            notes.insert(0, f"reissue of {reissue_of} on newer sales data, with the weather that run saved")
        else:
            rows, notes, last_known, weather = build_rows(now, trigger, run_id, log_repo, version, commit, also_run, with_holiday_rule)
        run_row["data_through"] = str(last_known.date())
        rows = logio.new_rows_only(rows, log_repo / "log" / "forecasts.csv")
        if not rows:
            run_row["status"] = "skipped"
            notes.append("no new forecast rows (already logged, or nothing due)")
        run_row["reason"] = "; ".join(notes)
        stale = (now.date() - last_known.date()).days
        if stale > 8:
            run_row["reason"] = (run_row["reason"] + f"; sales data is {stale} days old").strip("; ")
    except Exception as exc:  # the failure is itself a log line: silence must not hide it
        run_row["status"] = "aborted"
        run_row["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        rows, weather = [], []
    if dry_run:
        print(f"[dry run] {run_row['status']} {run_row['reason']}")
        _print_summary(rows)
        return 0 if run_row["status"] != "aborted" else 1
    logio.write_run(log_repo, rows, run_row, weather if rows else [])
    outcome = logio.commit_and_push(log_repo, f"run {run_id}: {run_row['status']} ({len(rows)} rows)")
    print(f"[run] {run_row['status']}: {len(rows)} rows, {outcome}. {run_row['reason']}")
    _print_summary(rows)
    return 0 if run_row["status"] != "aborted" else 1


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    table = frame.pivot_table(index=["forecast_type", "target_date"], columns=["model", "variant"], values="forecast", aggfunc="first")
    pd.set_option("display.width", 220)
    print(table.round(0).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trigger", required=True, choices=["monday", "evening", "NW", "TW", "TD"])
    parser.add_argument("--now", help="issue time (ISO, e.g. 2026-11-02T12:30:00+00:00); default: the current time")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reissue-of", metavar="RUN_ID", help="make this withdrawn run again on the sales data now available, with the days and weather it had")
    args = parser.parse_args()
    if args.reissue_of and args.now:
        parser.error("a reissue is made at the real time: --now cannot be given with --reissue-of")
    settings = load_settings()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    with RunLock():
        return run(args.trigger, now, args.dry_run, Path(settings["log_repo"]), str(settings["model_version"]), tuple(str(v) for v in (settings.get("also_run") or ())),
                   bool(settings.get("holiday_rule")), args.reissue_of)


if __name__ == "__main__":
    sys.exit(main())
