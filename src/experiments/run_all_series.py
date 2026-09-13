"""
Regenerate every result in docs/EXPERIMENT_LOG.md from committed code.

Series run in dependency order - several consume earlier series' saved
predictions:

    9 -> 10, 11 -> 12        (December corrections)
    14 -> 15, 16 -> 17       (log1p winner, ensemble, seed bagging,
                              manual-forecast comparison)
    17 -> business impact

Series 1-4 are skipped unless --include-1-4 is passed: Series 1's full
algorithm sweep fits every model family including SARIMAX and Prophet and
takes far longer than everything else combined. Their results are not
needed as inputs by any later series.

Usage:
    python -m src.experiments.run_all_series
    python -m src.experiments.run_all_series --include-1-4
    python -m src.experiments.run_all_series --only 14 15 16

Expects data/features/engineered_features.csv, produced by `python main.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

LATER_SERIES = [
    ("5", "run_series5_christmas_period"),
    ("6", "run_series6_december_intensity"),
    ("7", "run_series7_december_phases"),
    ("8", "run_series8_hyperparameter_retune"),
    ("9", "run_series9_multiplicative_correction"),
    ("10", "run_series10_clean_slate_correction"),
    ("11", "run_series11_calibration_ratio"),
    ("12", "run_series12_pooled_calibration"),
    ("13", "run_series13_mae_loss"),
    ("14", "run_series14_log1p_target"),
    ("15", "run_series15_prophet_ensemble"),
    ("16", "run_series16_seed_bagging"),
    ("17", "run_series17_manual_forecast_comparison"),
    ("business", "run_business_impact"),
]

# Series 1-4 were built with their own CLIs; run them as subprocesses with
# the arguments that reproduce the logged results.
EARLY_SERIES = [
    ("1", ["-m", "src.experiments.run_sweep", "--full"]),
    ("2", ["-m", "src.experiments.run_feature_sweep"]),
    ("3", ["-m", "src.experiments.recursive_horizon_test"]),
    ("4", ["-m", "src.experiments.run_series4_sweep"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-1-4", action="store_true", help="Also re-run Series 1-4 (slow).")
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these series ids, e.g. --only 14 15 16 business")
    args = parser.parse_args()

    selected = set(args.only) if args.only else None

    if args.include_1_4:
        for series_id, argv in EARLY_SERIES:
            if selected and series_id not in selected:
                continue
            print(f"\n===== Series {series_id} =====", flush=True)
            subprocess.run([sys.executable, *argv], check=True)

    for series_id, module_name in LATER_SERIES:
        if selected and series_id not in selected:
            continue
        print(f"\n===== Series {series_id} ({module_name}) =====", flush=True)
        started = time.time()
        subprocess.run([sys.executable, "-m", f"src.experiments.{module_name}"], check=True)
        print(f"----- finished in {time.time() - started:.0f}s -----", flush=True)


if __name__ == "__main__":
    main()
