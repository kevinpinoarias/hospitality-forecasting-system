"""
Tests for the shared file helpers and daily-build loop in
src/preprocessing/common.py.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.build_daily_sales import build_daily_from_week
from src.preprocessing.common import build_daily_dataset, build_file_list


def weekly_sales(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date of business": day, "Subdivision": "Actual", "Stream": "Food", "Value": 100},
            {"Date of business": day, "Subdivision": "Forecast", "Stream": "Food", "Value": 90},
        ]
    )


def build(input_dir, tmp_path) -> pd.DataFrame:
    out_dir = tmp_path / "out"
    per_file_dir = out_dir / "per_file"
    per_file_dir.mkdir(parents=True)
    return build_daily_dataset(
        label="sales",
        input_dir=input_dir,
        output_dir=out_dir,
        per_file_dir=per_file_dir,
        per_file_suffix="_daily.csv",
        master_name="master.csv",
        summary_name="summary.csv",
        build_week=build_daily_from_week,
    )


def test_file_list_keeps_csv_and_xlsx_and_skips_temp_and_hidden_files(tmp_path):
    for name in ["a.csv", "b.xlsx", "~$b.xlsx", ".manifest.json", "notes.txt"]:
        (tmp_path / name).write_text("x")
    assert [p.name for p in build_file_list(tmp_path)] == ["a.csv", "b.xlsx"]


def test_the_daily_build_reads_xlsx_weeks_as_well_as_csv(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    weekly_sales("02/01/2024").to_csv(input_dir / "week1_sales.csv", index=False)
    weekly_sales("09/01/2024").to_excel(input_dir / "week2_sales.xlsx", index=False)

    master = build(input_dir, tmp_path)

    assert sorted(master["source_file"]) == ["week1_sales.csv", "week2_sales.xlsx"]
    assert master["total_sales"].tolist() == [100, 100]
