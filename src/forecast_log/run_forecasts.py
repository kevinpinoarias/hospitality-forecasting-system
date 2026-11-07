"""
One run of the forecast log: make the forecasts that are due, append them to the log, commit and push (rules: PREREGISTRATION.md in the `forecasting-log` repository).

  python -m src.forecast_log.run_forecasts --trigger monday      next week (NW) and this week (TW)               scheduled Monday 12:30 (scripts/run_forecast_log.cmd weekly)
  python -m src.forecast_log.run_forecasts --trigger evening     tomorrow (TD) only                              by hand
  add --dry-run to compute and print everything without writing, committing or pushing

For every forecast type the run makes the three models (portfolio, post, hydro) without the manager's forecast (it only reaches us after the day; the stand-in takes its place), with the
weather the live API would use, and logs the seasonal baseline as a benchmark; the manager's own number is added afterwards, when the export brings it (score_forecasts.py). The weather the run
used is saved to log/weather_used.csv. A run issued after its deadline is labelled `late`. A run that cannot proceed (models changed, features missing, ...) writes an `aborted` line to runs.csv
with the reason and makes no forecast. Settings: config/forecast_log.yaml (the log repository and the model version to use), overridden by config/forecast_log.local.yaml (not tracked).
`also_run: [v1.11]` (a setting, off by default) issues a further registered version's hydro arm beside the main version, as rows named `hydro_v1.11`; the version must be registered and its models and
correction history present, or the run aborts and says so.
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
from src.forecast_log import inputs, logio, predict, schedule, v1
from src.models.final_model import DATE_COL, TARGET_COL

FEATURES = PROJECT_ROOT / "data" / "features" / "engineered_features.csv"
LOCK = PROJECT_ROOT / ".forecast_log.lock"
STALE_LOCK_SECONDS = 2 * 3600


def load_settings() -> dict:
    settings = {"log_repo": r"D:\Projectos\DRG\forecasting-log", "model_version": v1.VERSION}
    for name in ("forecast_log.yaml", "forecast_log.local.yaml"):
        path = PROJECT_ROOT / "config" / name
        if path.exists():
            settings.update({k: v for k, v in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items() if v is not None})
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


def weather_rows(run_id: str, weather: pd.DataFrame) -> list[dict]:
    fetched = logio.stamp()
    return [{"run_id": run_id, "date": d.strftime("%Y-%m-%d"), "max_temp": t, "rain_mm": r, "sun_hours": s, "fetched_at": fetched}
            for d, t, r, s in zip(weather[DATE_COL], weather["max_temp"], weather["rain_mm"], weather["sun_hours"])]


def build_rows(issue: datetime, trigger: str, run_id: str, log_repo: Path, version: str, commit: str,
               also_run: tuple[str, ...] = ()) -> tuple[list[dict], list[str], pd.Timestamp, list[dict]]:
    """The log rows this run would write, notes about what was skipped, the date the sales data runs through, and the weather the run used.
    `also_run` lists further registered versions (v1.11) whose hydro arm is issued beside the main version's three models, as rows of their own."""
    full = h.load_engineered(FEATURES)
    known = full.dropna(subset=[TARGET_COL])
    last_known = known[DATE_COL].max()
    models = predict.load_models(log_repo, version)
    history = v1.load_correction_history(v1.correction_history_path(version))
    extras = [(m, v1.load_correction_history(v1.correction_history_path(m.version))) for m in (predict.load_models(log_repo, v) for v in also_run)]
    features_sha = h.file_sha256(FEATURES)
    weather, errored = predict.fetch_run_weather(last_known, issue.date())
    through = inputs.hydro_list_through()
    rows: list[dict] = []
    notes: list[str] = ["weather service unavailable: historical averages used"] if errored else []
    for ftype in schedule.forecast_types_for(trigger):
        targets = [d for d in schedule.target_dates(ftype, issue) if pd.Timestamp(d) > last_known]  # never forecast a day with a known actual
        if not targets:
            notes.append(f"{ftype}: no day to forecast (all started or already known)")
            continue
        label = "on_time" if schedule.on_time(ftype, issue) else "late"
        unknown = inputs.unknown_hydro_thursdays(targets, through)
        if unknown:
            notes.append(f"{ftype}: hydro model skipped on {sorted(str(d) for d in unknown)} (the Hydro list reaches {through})")
        base = {"run_id": run_id, "issue_time": issue.isoformat(timespec="seconds"), "forecast_type": ftype, "model_version": models.version, "data_through": str(last_known.date()),
                "hydro_list_through": str(through), "code_commit": commit, "on_time": label, "input_hash": input_hash(features_sha, weather), "manager_arrived_at": ""}
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
                rows.append({**common, "model": arm, "model_version": extra.version, "forecast": round(float(result[arm].iloc[i]), 2), "model_fingerprint": extra.combined_fingerprint})
        # the benchmark that needs no model, once per forecast type and day (the manager's number is added by the scoring once the export brings it)
        seasonal = predict.seasonal_baseline(full, last_known, [pd.Timestamp(d) for d in targets])
        for i, d in enumerate(targets):
            rows.append({**base, "target_date": str(d), "lead_days": (d - issue.date()).days, "variant": "", "weather_source": "", "model": "seasonal",
                         "forecast": round(float(seasonal[i]), 2), "manager_forecast_used": "", "model_fingerprint": ""})
    return rows, notes, last_known, weather_rows(run_id, weather)


def run(trigger: str, now: datetime, dry_run: bool, log_repo: Path, version: str, also_run: tuple[str, ...] = ()) -> int:
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
        rows, notes, last_known, weather = build_rows(now, trigger, run_id, log_repo, version, commit, also_run)
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
    args = parser.parse_args()
    settings = load_settings()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    with RunLock():
        return run(args.trigger, now, args.dry_run, Path(settings["log_repo"]), str(settings["model_version"]), tuple(str(v) for v in (settings.get("also_run") or ())))


if __name__ == "__main__":
    sys.exit(main())
