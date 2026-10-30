"""
One run of the forecast log: make the forecasts that are due, append them to the log, commit and push (rules: PREREGISTRATION.md in the `forecasting-log` repository).

  python -m src.forecast_log.run_forecasts --trigger monday      next week (NW) and this week (TW)        scheduled Monday 08:00
  python -m src.forecast_log.run_forecasts --trigger evening     tomorrow (TD)                            scheduled every evening 18:00
  add --dry-run to compute and print everything without writing, committing or pushing

For every forecast type the run makes the three models (portfolio, post, hydro) WITHOUT the manager's forecast, always, and WITH it for the days whose manager number is in the manager
folder when the run starts; the manager's own number and the seasonal baseline are logged as benchmarks. A run issued after its deadline is labelled `late`. A run that cannot proceed (models
changed, features missing, ...) writes an `aborted` line to runs.csv with the reason and makes no forecast. Settings: config/forecast_log.yaml, overridden by config/forecast_log.local.yaml
(not tracked: use it for the manager folder).
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
    settings = {"log_repo": r"D:\Projectos\DRG\forecasting-log", "manager_dir": None}
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


def input_hash(features_sha: str, manager_files: list[str]) -> str:
    return hashlib.sha256((features_sha + "|" + ",".join(sorted(manager_files))).encode()).hexdigest()[:32]


def build_rows(issue: datetime, trigger: str, run_id: str, log_repo: Path, manager_dir: Path | None, commit: str) -> tuple[list[dict], list[str], pd.Timestamp]:
    """The log rows this run would write, notes about what was skipped, and the date the sales data runs through."""
    full = h.load_engineered(FEATURES)
    known = full.dropna(subset=[TARGET_COL])
    last_known = known[DATE_COL].max()
    models = predict.load_models(log_repo)
    history = v1.load_correction_history()
    features_sha = h.file_sha256(FEATURES)
    manager_all = inputs.read_manager_forecasts(manager_dir)
    through = inputs.hydro_list_through()
    rows: list[dict] = []
    notes: list[str] = []
    for ftype in schedule.forecast_types_for(trigger):
        targets = [d for d in schedule.target_dates(ftype, issue) if pd.Timestamp(d) > last_known]  # never forecast a day with a known actual
        if not targets:
            notes.append(f"{ftype}: no day to forecast (all started or already known)")
            continue
        label = "on_time" if schedule.on_time(ftype, issue) else "late"
        unknown = inputs.unknown_hydro_thursdays(targets, through)
        if unknown:
            notes.append(f"{ftype}: hydro model skipped on {sorted(str(d) for d in unknown)} (the Hydro list reaches {through})")
        mgr = inputs.manager_for(manager_all, targets, not_after=issue)
        runs = [("no_manager", targets, None, [])]
        if not mgr.empty:
            runs.append(("with_manager", [d.date() for d in mgr["date"]], mgr, sorted(mgr["file"].unique())))
        elif manager_dir is None:
            notes.append(f"{ftype}: no manager folder configured; without-manager rows only")
        else:
            notes.append(f"{ftype}: no manager forecast in the folder for these days; without-manager rows only")
        base = {"run_id": run_id, "issue_time": issue.isoformat(timespec="seconds"), "forecast_type": ftype, "model_version": models.version,
                "data_through": str(last_known.date()), "weather_source": predict.WEATHER_SOURCE, "hydro_list_through": str(through), "code_commit": commit, "on_time": label}
        for variant, days, mgr_rows, files in runs:
            ts = [pd.Timestamp(d) for d in days]
            manager_arr = None if mgr_rows is None else mgr_rows["forecast"].to_numpy()
            result = predict.forecast_days(models, full, last_known, ts, manager_arr, history)
            for i, d in enumerate(days):
                common = {**base, "target_date": str(d), "lead_days": (d - issue.date()).days, "variant": variant, "input_hash": input_hash(features_sha, files),
                          "manager_forecast_used": round(float(result["forecast_sales_input"].iloc[i]), 2),
                          "manager_arrived_at": "" if mgr_rows is None else mgr_rows["arrived_at"].iloc[i].isoformat(timespec="seconds")}
                for model in predict.MODELS:
                    if model == "hydro" and d in unknown:
                        continue
                    fingerprint = models.portfolio_fingerprint if model == "portfolio" else models.combined_fingerprint
                    rows.append({**common, "model": model, "forecast": round(float(result[model].iloc[i]), 2), "model_fingerprint": fingerprint})
        # benchmarks, once per forecast type and day
        bench = {**base, "variant": "", "input_hash": input_hash(features_sha, []), "model_fingerprint": "", "manager_arrived_at": ""}
        seasonal = predict.seasonal_baseline(full, last_known, [pd.Timestamp(d) for d in targets])
        for i, d in enumerate(targets):
            rows.append({**bench, "target_date": str(d), "lead_days": (d - issue.date()).days, "model": "seasonal", "forecast": round(float(seasonal[i]), 2), "manager_forecast_used": ""})
        if not mgr.empty:
            for _, m in mgr.iterrows():
                rows.append({**bench, "target_date": str(m["date"].date()), "lead_days": (m["date"].date() - issue.date()).days, "model": "manager", "forecast": round(float(m["forecast"]), 2),
                             "manager_forecast_used": round(float(m["forecast"]), 2), "manager_arrived_at": m["arrived_at"].isoformat(timespec="seconds")})
    return rows, notes, last_known


def run(trigger: str, now: datetime, dry_run: bool, log_repo: Path, manager_dir: Path | None) -> int:
    run_id = f"{now:%Y%m%dT%H%M}-{trigger}"
    commit = v1.code_commit()
    types = "+".join(schedule.forecast_types_for(trigger))
    run_row = {"run_id": run_id, "issue_time": now.isoformat(timespec="seconds"), "trigger": trigger, "status": "ok", "reason": "", "data_through": "", "forecast_types_made": types,
               "code_commit": commit}
    rows: list[dict] = []
    try:
        if not dry_run:
            pending = logio.push_pending(log_repo)
            if pending == "push_failed":
                print("[run] warning: an earlier commit could not be pushed; will retry after this run")
        rows, notes, last_known = build_rows(now, trigger, run_id, log_repo, manager_dir, commit)
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
        rows = []
    if dry_run:
        print(f"[dry run] {run_row['status']} {run_row['reason']}")
        _print_summary(rows)
        return 0 if run_row["status"] != "aborted" else 1
    logio.write_run(log_repo, rows, run_row)
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
    parser.add_argument("--now", help="issue time (ISO, e.g. 2026-11-02T08:00:00+00:00); default: the current time")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    manager_dir = Path(settings["manager_dir"]) if settings.get("manager_dir") else None
    with RunLock():
        return run(args.trigger, now, args.dry_run, Path(settings["log_repo"]), manager_dir)


if __name__ == "__main__":
    sys.exit(main())
