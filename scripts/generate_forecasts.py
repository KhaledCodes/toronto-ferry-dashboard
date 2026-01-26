"""
Script to generate forecasts using trained models.
Usage: python scripts/generate_forecasts.py [--long-term]
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_inference import (
    generate_forecasts, create_daily_summary,
    generate_long_term_forecasts
)
from src.config import OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description='Generate ferry ridership forecasts')
    parser.add_argument('--long-term', action='store_true',
                        help='Also generate 365-day long-term forecasts')
    parser.add_argument('--only-long-term', action='store_true',
                        help='Only generate long-term forecasts (skip 7-day)')
    args = parser.parse_args()

    print("="*60)
    print(" Toronto Ferry Ridership Forecast Generation")
    print("="*60)

    # Generate 7-day forecasts (unless only-long-term is specified)
    if not args.only_long_term:
        print("\n--- 7-Day Forecast ---")
        forecasts = generate_forecasts()

        if len(forecasts) == 0:
            print("No 7-day forecasts generated.")
        else:
            # Create daily summary
            print("\nCreating daily summary...")
            daily = create_daily_summary(forecasts)

            # Save daily summary
            daily_path = OUTPUTS_DIR / "daily_forecasts.csv"
            daily.to_csv(daily_path, index=False)
            print(f"Daily summary saved to {daily_path}")

            # Print daily forecast table
            print("\n" + "="*60)
            print(" 7-Day Forecast Summary")
            print("="*60)
            print(f"{'Date':<12} {'Redemptions':>12} {'Sales':>10} {'Temp':>8} {'Precip':>8}")
            print("-"*60)

            for _, row in daily.iterrows():
                print(f"{str(row['date']):<12} {row['total_redemptions']:>12,} {row['total_sales']:>10,} "
                      f"{row['avg_temp']:>7.1f}C {row['total_precip']:>7.1f}mm")

            print("="*60)

    # Generate long-term forecasts if requested
    if args.long_term or args.only_long_term:
        print("\n" + "="*60)
        print(" Long-Term (365-Day) Forecast Generation")
        print("="*60)

        long_term_forecasts = generate_long_term_forecasts()

        if len(long_term_forecasts) == 0:
            print("No long-term forecasts generated.")
            print("Make sure no-weather models are trained (run train_models.py --no-weather)")
        else:
            # Print monthly summary
            print("\n" + "="*60)
            print(" Monthly Forecast Summary")
            print("="*60)
            print(f"{'Month':<10} {'Redemptions':>15} {'Sales':>12} {'Confidence':>12}")
            print("-"*60)

            # Read monthly summary
            monthly_path = OUTPUTS_DIR / "long_term_monthly_forecasts.csv"
            if monthly_path.exists():
                import pandas as pd
                monthly = pd.read_csv(monthly_path)
                for _, row in monthly.iterrows():
                    print(f"{row['year_month']:<10} {row['total_redemptions']:>15,} "
                          f"{row['total_sales']:>12,} {row['forecast_confidence']:>12}")

            print("="*60)


if __name__ == "__main__":
    main()
