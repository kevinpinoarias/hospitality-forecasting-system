"""
Tests for src/ingestion/anonymize_raw.py.

These exist specifically to guard the boundary between the private raw
landing directory (real business-identifying data) and the tracked raw
directory (what actually enters git and the rest of the pipeline) - the
single most important property of this script is that identifying
information never crosses that boundary, even as the source export format
evolves.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ingestion.anonymize_raw import (
    OUTPUT_COLUMNS,
    anonymise_dataframe,
    anonymize_raw,
    normalise_department,
)

RAW_COLUMNS = [
    "Site",
    "Department",
    "Metric",
    "Subdivision",
    "Date of business",
    "UTC datetime",
    "Stream",
    "Value",
]


def make_raw_df(**overrides) -> pd.DataFrame:
    row = {
        "Site": "Example Venue City",
        "Department": "ABC (FOH)",
        "Metric": "Sales",
        "Subdivision": "Forecast",
        "Date of business": "02/01/2024",
        "UTC datetime": "02/01/2024 00:00:00",
        "Stream": "Food",
        "Value": 6000,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_site_column_is_dropped():
    out = anonymise_dataframe(make_raw_df())
    assert "Site" not in out.columns


def test_department_prefix_is_stripped():
    out = anonymise_dataframe(make_raw_df(Department="ABC (Kitchen)"))
    assert out["Department"].iloc[0] == "Kitchen"


def test_output_columns_are_exactly_the_allow_list():
    out = anonymise_dataframe(make_raw_df())
    assert list(out.columns) == OUTPUT_COLUMNS


def test_unexpected_extra_column_is_dropped_not_leaked():
    """If a future export adds a new column (e.g. a manager name), it must
    never survive into the output - the allow-list must win even for
    columns this script has never seen before."""
    df = make_raw_df()
    df["Manager Name"] = "Someone Identifiable"
    out = anonymise_dataframe(df)
    assert "Manager Name" not in out.columns
    assert list(out.columns) == OUTPUT_COLUMNS


def test_site_value_never_appears_anywhere_in_output():
    out = anonymise_dataframe(make_raw_df())
    for col in out.columns:
        assert not out[col].astype(str).str.contains("Example Venue City").any()


def test_missing_required_column_raises():
    df = make_raw_df().drop(columns=["Metric"])
    with pytest.raises(ValueError, match="missing expected column"):
        anonymise_dataframe(df)


def test_unrecognised_department_shape_raises_rather_than_passing_through():
    """A department value that doesn't match 'CODE (Name)' might itself be
    identifying (e.g. a raw employee-facing label) - it must never be
    passed through unchanged."""
    with pytest.raises(ValueError, match="did not match the expected"):
        anonymise_dataframe(make_raw_df(Department="Front of House"))


def test_normalise_department_handles_multiple_venue_codes():
    """Regex-based, not a hardcoded per-venue mapping - must keep working
    if this is ever run against a different site's department codes."""
    result = normalise_department(pd.Series(["ABC (FOH)", "XYZ99 (Bar)"]))
    assert list(result) == ["FOH", "Bar"]


# ---------------------------------------------------------------------
# End-to-end incremental behaviour
# ---------------------------------------------------------------------

@pytest.fixture
def private_and_output_dirs(tmp_path):
    input_dir = tmp_path / "raw_private"
    output_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_dir.mkdir()
    manifest_path = input_dir / ".anonymize_manifest.json"
    return input_dir, output_dir, manifest_path


def write_raw_csv(path, **overrides):
    make_raw_df(**overrides).to_csv(path, index=False)


def test_run_writes_anonymised_output_and_manifest(private_and_output_dirs):
    input_dir, output_dir, manifest_path = private_and_output_dirs
    write_raw_csv(input_dir / "week01.csv")

    anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)

    out_df = pd.read_csv(output_dir / "week01.csv")
    assert "Site" not in out_df.columns
    assert out_df["Department"].iloc[0] == "FOH"
    assert manifest_path.exists()


def test_second_run_skips_already_processed_files(private_and_output_dirs):
    input_dir, output_dir, manifest_path = private_and_output_dirs
    write_raw_csv(input_dir / "week01.csv")

    anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)
    summary = anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)

    assert (summary["status"] == "SKIP_ALREADY_PROCESSED").all()


def test_new_file_is_processed_without_reprocessing_existing_ones(private_and_output_dirs):
    input_dir, output_dir, manifest_path = private_and_output_dirs
    write_raw_csv(input_dir / "week01.csv")
    anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)

    write_raw_csv(input_dir / "week02.csv", Department="ABC (Kitchen)")
    summary = anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)

    statuses = dict(zip(summary["file"], summary["status"]))
    assert statuses["week01.csv"] == "SKIP_ALREADY_PROCESSED"
    assert statuses["week02.csv"] == "OK"


def test_changed_file_content_is_reprocessed(private_and_output_dirs):
    """A source file with the same name but different content (e.g. a
    corrected re-export) should be reprocessed, not skipped - the manifest
    keys on content hash, not just filename."""
    input_dir, output_dir, manifest_path = private_and_output_dirs
    write_raw_csv(input_dir / "week01.csv", Value=6000)
    anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)

    write_raw_csv(input_dir / "week01.csv", Value=9999)
    summary = anonymize_raw(input_dir=input_dir, output_dir=output_dir, manifest_path=manifest_path)

    assert (summary["status"] == "OK").all()
    out_df = pd.read_csv(output_dir / "week01.csv")
    assert out_df["Value"].iloc[0] == 9999


def test_missing_input_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        anonymize_raw(
            input_dir=tmp_path / "does_not_exist",
            output_dir=tmp_path / "raw",
            manifest_path=tmp_path / "manifest.json",
        )
