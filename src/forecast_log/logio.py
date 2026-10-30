"""
Writing to the forecast log (the `forecasting-log` repository): append-only rows, one line per run, commit and push.

Rows are only ever appended. A forecast is not repeated for a (forecast type, target date, model, variant) that already has a row: the first forecast counts, a second run for the same
period adds nothing. The push is what timestamps a run, so the run commits and pushes at once; if the push fails the commit stays local and is pushed by the next run (the run says so).
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

FORECAST_COLUMNS = ["run_id", "issue_time", "target_date", "lead_days", "forecast_type", "model", "model_version", "variant", "forecast", "manager_forecast_used",
                    "manager_arrived_at", "data_through", "weather_source", "hydro_list_through", "input_hash", "model_fingerprint", "code_commit", "on_time"]
RUN_COLUMNS = ["run_id", "issue_time", "trigger", "status", "reason", "data_through", "forecast_types_made", "code_commit"]


def existing_keys(forecasts_csv: Path) -> set[tuple[str, str, str, str]]:
    """(forecast type, target date, model, variant) of every row already in the log."""
    if not forecasts_csv.exists():
        return set()
    frame = pd.read_csv(forecasts_csv, dtype=str)
    if frame.empty:
        return set()
    return set(zip(frame["forecast_type"], frame["target_date"], frame["model"], frame["variant"].fillna("")))


def new_rows_only(rows: list[dict], forecasts_csv: Path) -> list[dict]:
    seen = existing_keys(forecasts_csv)
    out = []
    for r in rows:
        key = (r["forecast_type"], str(r["target_date"]), r["model"], r["variant"] or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _append(path: Path, columns: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    ensure_header(path, columns)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        for r in rows:
            writer.writerow(r)


def ensure_header(path: Path, columns: list[str]) -> None:
    """The file must start with exactly these columns (the header may only have been created by the repository's first commit)."""
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(",".join(columns) + "\n", encoding="utf-8")
        return
    first = path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first != ",".join(columns):
        raise RuntimeError(f"{path.name} has a different header ({first}) from the one this code writes")


def write_run(log_repo: Path, forecast_rows: list[dict], run_row: dict) -> None:
    _append(log_repo / "log" / "forecasts.csv", FORECAST_COLUMNS, forecast_rows)
    _append(log_repo / "log" / "runs.csv", RUN_COLUMNS, [run_row])


def git(log_repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=log_repo, capture_output=True, text=True, check=check)


def commit_and_push(log_repo: Path, message: str) -> str:
    """Commit the log files and push. Returns 'pushed', 'push_failed' (committed locally) or 'nothing_to_commit'."""
    git(log_repo, "add", "log", "models")
    if not git(log_repo, "status", "--porcelain").stdout.strip():
        return "nothing_to_commit"
    git(log_repo, "commit", "-q", "-m", message)
    pushed = git(log_repo, "push", "origin", "main", check=False)
    return "pushed" if pushed.returncode == 0 else "push_failed"


def push_pending(log_repo: Path) -> str:
    """Push commits a previous run could not push (before this run's own work). Returns 'none', 'pushed' or 'push_failed'."""
    ahead = git(log_repo, "rev-list", "--count", "origin/main..HEAD", check=False).stdout.strip()
    if ahead in ("", "0"):
        return "none"
    return "pushed" if git(log_repo, "push", "origin", "main", check=False).returncode == 0 else "push_failed"


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
