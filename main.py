"""
Main entry point for the hospitality forecasting portfolio project.

Runs the full pipeline:
1. Split raw files into sales and labour
2. Build daily sales dataset
3. Build daily labour dataset
4. Build engineered and model-ready features
5. Run baseline models
6. Evaluate human forecast
7. Train SARIMAX
8. Train XGBoost
"""

from __future__ import annotations

from src.preprocessing.split_sales_labour import split_sales_and_labour
from src.preprocessing.build_daily_sales import build_daily_sales
from src.preprocessing.build_daily_labour import build_daily_labour
from src.features.build_features import build_feature_dataset
from src.baselines.run_baselines import run_baselines
from src.models.evaluate_human_forecast import evaluate_human_forecast
from src.models.train_sarimax import run_sarimax
from src.models.train_xgboost import run_xgboost


def main() -> None:
    print("\n[1/8] Splitting raw sales and labour files...")
    split_sales_and_labour()

    print("\n[2/8] Building daily sales dataset...")
    build_daily_sales()

    print("\n[3/8] Building daily labour dataset...")
    build_daily_labour()

    print("\n[4/8] Building feature datasets...")
    build_feature_dataset()

    print("\n[5/8] Running baseline models...")
    run_baselines()

    print("\n[6/8] Evaluating human forecast...")
    evaluate_human_forecast()

    print("\n[7/8] Training SARIMAX...")
    run_sarimax()

    print("\n[8/8] Training XGBoost...")
    run_xgboost()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()