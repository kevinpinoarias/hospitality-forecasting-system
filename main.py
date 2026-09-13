"""
Main entry point for the hospitality forecasting portfolio project.

Runs the full pipeline:
1. Anonymise fresh raw exports
2. Split raw files into sales and labour
3. Build daily sales dataset
4. Build daily labour dataset
5. Build engineered and model-ready features
6. Run baseline models
7. Evaluate human forecast
8. Train SARIMAX
9. Train XGBoost
10. Train the final model (the one the API serves)
"""

from __future__ import annotations

from src.ingestion.anonymize_raw import INPUT_DIR as RAW_PRIVATE_DIR, anonymize_raw, has_raw_exports
from src.preprocessing.split_sales_labour import split_sales_and_labour
from src.preprocessing.build_daily_sales import build_daily_sales
from src.preprocessing.build_daily_labour import build_daily_labour
from src.features.build_features import build_feature_dataset
from src.baselines.run_baselines import run_baselines
from src.models.evaluate_human_forecast import evaluate_human_forecast
from src.models.train_sarimax import run_sarimax
from src.models.train_xgboost import run_xgboost
from src.models.train_final_model import run_final_model


def main() -> None:
    print("\n[1/10] Anonymising fresh raw exports...")
    if has_raw_exports():
        anonymize_raw()
    else:
        print(f"No raw exports in {RAW_PRIVATE_DIR} - skipping; the tracked data in data/raw/ is already anonymised.")

    print("\n[2/10] Splitting raw sales and labour files...")
    split_sales_and_labour()

    print("\n[3/10] Building daily sales dataset...")
    build_daily_sales()

    print("\n[4/10] Building daily labour dataset...")
    build_daily_labour()

    print("\n[5/10] Building feature datasets...")
    build_feature_dataset()

    print("\n[6/10] Running baseline models...")
    run_baselines()

    print("\n[7/10] Evaluating human forecast...")
    evaluate_human_forecast()

    print("\n[8/10] Training SARIMAX...")
    run_sarimax()

    print("\n[9/10] Training XGBoost...")
    run_xgboost()

    print("\n[10/10] Training the final model...")
    run_final_model()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()