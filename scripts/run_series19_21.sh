#!/usr/bin/env bash
# Runs Series 20, 21 and 19 in turn (Series 22, which needs Series 20, is run with
# `python -m src.experiments.run_series22_neural_target_day`) (each resumable: finished tasks are skipped on a re-run).
set -u
cd "$(dirname "$0")/.."   # run from the repository root
export PYTHONPATH=. WANDB_MODE=disabled PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
for s in run_series20_neural_v2 run_series21_catboost_depth_iterations run_series19_column_order; do
  echo "===== $s started $(date) ====="
  python -m src.experiments.$s --workers 10
  echo "===== $s finished $(date) (exit $?) ====="
done
echo "ALL DONE $(date)"
