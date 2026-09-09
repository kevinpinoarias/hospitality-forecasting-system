"""
Anonymises fresh raw operational exports before they enter the pipeline.

Purpose
-------
Fresh weekly exports land in data/raw_private/ (gitignored, never committed)
and contain real business-identifying information: a `Site` column naming
the venue, and `Department` values that embed a venue-specific brand code
(e.g. "ABC (FOH)"). This script strips both and writes the result to
data/raw/, which is the only raw data that ever enters version control or
the rest of the pipeline (src/preprocessing/split_sales_labour.py onward).

Design
------
- Allow-list, not block-list: the output schema is defined explicitly
  (OUTPUT_COLUMNS). Any input column not on that list is dropped rather
  than assumed harmless - if a future export adds a new identifying column
  (e.g. a manager name), this fails safe by never letting it through,
  instead of silently passing it downstream because nobody updated a
  block-list.
- Department is genericised with a regex, not a hardcoded per-venue
  mapping, so this keeps working unchanged if this system is ever run
  against a different site with different department codes. A department
  value that doesn't match the expected "CODE (Name)" shape is treated as
  an error, not passed through as-is - an unrecognised shape might itself
  be identifying.
- Incremental and idempotent: a manifest (data/raw_private/
  .anonymize_manifest.json) records which source files (by name + content
  hash) have already been processed, so re-running after a new weekly
  export lands only processes what's new.

Outputs
-------
- data/raw/*.csv                                  (anonymised)
- data/raw_private/.anonymize_manifest.json        (processing record)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw_private"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = INPUT_DIR / ".anonymize_manifest.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

EXTS = {".csv", ".xlsx"}

# The exact schema the rest of the pipeline expects (see
# src/preprocessing/split_sales_labour.py, build_daily_sales.py,
# build_daily_labour.py). Anything not in this list is dropped from the
# output, whether or not this script currently knows what it is.
OUTPUT_COLUMNS = [
    "Department",
    "Metric",
    "Subdivision",
    "Date of business",
    "UTC datetime",
    "Stream",
    "Value",
]

# A raw Department value looks like "ABC (FOH)" - a venue-specific brand
# code, then the generic department name in brackets. Only the bracketed
# part is retained.
DEPARTMENT_PATTERN = re.compile(r"^[A-Za-z0-9]+\s*\((.+)\)$")


# ---------------------------------------------------------------------
# Manifest (incremental processing state)
# ---------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def write_output(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)


def build_file_list(input_dir: Path) -> list[Path]:
    """Return candidate raw export files from the private landing directory."""
    return sorted(
        [
            p for p in input_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTS
            and not p.name.startswith("~$")
            and not p.name.startswith(".")
        ]
    )


def normalise_department(series: pd.Series) -> pd.Series:
    """Strip the venue-specific brand-code prefix from Department values.

    Raises if any value doesn't match the expected "CODE (Name)" shape,
    rather than passing an unrecognised value through unchanged - an
    unexpected shape might itself carry identifying information.
    """
    stripped = series.astype(str).str.strip()
    extracted = stripped.str.extract(DEPARTMENT_PATTERN, expand=False)

    unmatched = stripped[extracted.isna()].unique()
    if len(unmatched):
        raise ValueError(
            "Department value(s) did not match the expected 'CODE (Name)' "
            f"shape and cannot be safely anonymised: {sorted(unmatched)}"
        )

    return extracted


def anonymise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the anonymisation transform to one raw export's worth of rows."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw export is missing expected column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    df["Department"] = normalise_department(df["Department"])

    # Allow-list: keep only the columns the pipeline is meant to see, in a
    # fixed order, regardless of what else the source file contains.
    return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def anonymize_raw(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    """
    Anonymise every not-yet-processed file in the private raw landing
    directory and write the result to the tracked raw data directory.

    Returns
    -------
    pd.DataFrame
        Summary of processing results for this run.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Private raw landing directory not found: {input_dir}\n"
            "Fresh unanonymised exports must be placed there before running "
            "the pipeline - see src/ingestion/anonymize_raw.py."
        )

    manifest = load_manifest(manifest_path)
    files = build_file_list(input_dir)

    if not files:
        raise FileNotFoundError(f"No supported raw export files found in: {input_dir}")

    summary: list[dict] = []

    for path in files:
        file_hash = _file_hash(path)
        record = manifest.get(path.name)

        if record is not None and record.get("sha256") == file_hash:
            summary.append({"file": path.name, "status": "SKIP_ALREADY_PROCESSED"})
            continue

        try:
            df = read_file(path)
            clean = anonymise_dataframe(df)

            output_path = output_dir / path.name
            write_output(clean, output_path)

            try:
                output_path_display = str(output_path.relative_to(PROJECT_ROOT))
            except ValueError:
                output_path_display = str(output_path)

            manifest[path.name] = {
                "sha256": file_hash,
                "rows": len(clean),
                "output_path": output_path_display,
            }

            print(f"[OK] {path.name}: {len(df)} rows -> {output_path.name}")
            summary.append(
                {"file": path.name, "status": "OK", "rows_in": len(df), "rows_out": len(clean)}
            )

        except Exception as exc:
            print(f"[ERROR] {path.name}: {exc}")
            summary.append({"file": path.name, "status": "ERROR", "error": repr(exc)})

    save_manifest(manifest, manifest_path)

    summary_df = pd.DataFrame(summary)
    errors = summary_df[summary_df["status"] == "ERROR"] if len(summary_df) else summary_df
    if len(errors):
        raise RuntimeError(
            f"{len(errors)} raw file(s) failed anonymisation and were not written to "
            f"{output_dir}: {errors['file'].tolist()}"
        )

    n_ok = int((summary_df["status"] == "OK").sum()) if len(summary_df) else 0
    n_skipped = int((summary_df["status"] == "SKIP_ALREADY_PROCESSED").sum()) if len(summary_df) else 0
    print(f"\nAnonymised {n_ok} new file(s), skipped {n_skipped} already-processed file(s).")
    print(f"Manifest written to: {manifest_path}")

    return summary_df


if __name__ == "__main__":
    anonymize_raw()
