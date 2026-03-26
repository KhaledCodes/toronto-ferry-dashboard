"""
Script to train ferry ridership forecasting models.
Usage: python scripts/train_models.py [--no-weather] [--all]
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.config import FEATURES_CSV
from src.model_training import train_both_models, train_no_weather_models
from src.evaluation import print_metrics


def main():
    parser = argparse.ArgumentParser(description='Train ferry ridership forecasting models')
    parser.add_argument('--no-weather', action='store_true',
                        help='Train no-weather models for long-term forecasting')
    parser.add_argument('--all', action='store_true',
                        help='Train both weather and no-weather models')
    parser.add_argument('--only-no-weather', action='store_true',
                        help='Only train no-weather models (skip weather models)')
    parser.add_argument('--recent-only', action='store_true',
                        help='Train only on recent years (2022+), excluding COVID and flooding years')
    args = parser.parse_args()

    print("="*60)
    print(" Toronto Ferry Ridership Model Training")
    print("="*60)

    # Load data
    print(f"\nLoading data from {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} rows")

    # Filter to recent years if requested
    if args.recent_only:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['year'] = df['timestamp'].dt.year
        elif 'year' not in df.columns:
            print("Warning: Cannot filter by year - no timestamp or year column found")

        if 'year' in df.columns:
            original_rows = len(df)
            df = df[df['year'] >= 2022].copy()
            print(f"Filtered to years 2022+: {len(df):,} rows (removed {original_rows - len(df):,} rows)")

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'timestamp_hour' in df.columns:
        df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])

    # Train weather models (default behavior)
    if not args.only_no_weather:
        print("\n" + "="*60)
        print(" Training Weather Models (for 7-day forecasts)")
        print("="*60)
        redemption_model, sales_model, metrics = train_both_models(df)

        # Print detailed metrics
        print_metrics(metrics['redemption'], "Redemption Model Test Metrics")
        print_metrics(metrics['sales'], "Sales Model Test Metrics")

        # Print feature importance
        print("\n" + "="*60)
        print(" Top 15 Features (Redemption Model)")
        print("="*60)
        if redemption_model.feature_importance is not None:
            top_features = redemption_model.feature_importance.head(15)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']:40s} {row['importance']:10.0f}")

    # Train no-weather models if requested
    if args.no_weather or args.all or args.only_no_weather:
        print("\n" + "="*60)
        print(" Training No-Weather Models (for long-term forecasts)")
        print("="*60)
        redemption_nw, sales_nw, nw_metrics = train_no_weather_models(df)

        # Print detailed metrics
        print_metrics(nw_metrics['redemption_no_weather'], "No-Weather Redemption Model Test Metrics")
        print_metrics(nw_metrics['sales_no_weather'], "No-Weather Sales Model Test Metrics")

        # Print feature importance for no-weather model
        print("\n" + "="*60)
        print(" Top 15 Features (No-Weather Redemption Model)")
        print("="*60)
        if redemption_nw.feature_importance is not None:
            top_features = redemption_nw.feature_importance.head(15)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']:40s} {row['importance']:10.0f}")

    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
