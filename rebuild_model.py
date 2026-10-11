"""
Rebuild the model the API serves, from the tracked raw data, and nothing else.

This is what CI runs. It is steps 2-5 and 10 of main.py - split the raw
files, build the daily sales and labour tables, build the features, train
the final model - and skips the baselines, SARIMAX and XGBoost steps, which
the API does not need. About 90 seconds in total.

Run it locally with:   python rebuild_model.py
(set WANDB_MODE=disabled if you are not logged in to Weights & Biases)

If you change the steps in main.py, change them here too.
"""

from __future__ import annotations

import time

from src.features.build_features import build_feature_dataset
from src.models.train_final_model import run_final_model
from src.preprocessing.build_daily_labour import build_daily_labour
from src.preprocessing.build_daily_sales import build_daily_sales
from src.preprocessing.split_sales_labour import split_sales_and_labour

STEPS = [
    ("Splitting raw sales and labour files", split_sales_and_labour),
    ("Building daily sales dataset", build_daily_sales),
    ("Building daily labour dataset", build_daily_labour),
    ("Building feature datasets (downloads weather)", build_feature_dataset),
    ("Training the final model", run_final_model),
]


def main() -> None:
    started = time.time()
    for number, (description, step) in enumerate(STEPS, start=1):
        print(f"\n[{number}/{len(STEPS)}] {description}...", flush=True)
        step_started = time.time()
        step()
        print(f"    done in {time.time() - step_started:.0f}s", flush=True)
    print(f"\nRebuild finished in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    main()
