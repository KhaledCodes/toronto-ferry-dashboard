"""
Aggregate 15-minute ferry ticket data to daily and hourly totals for the dashboard.
"""
import pandas as pd
from pathlib import Path

outputs = Path(__file__).parent.parent / "outputs"

df = pd.read_csv(outputs / "ferry_ticket_counts.csv", parse_dates=["Timestamp"])

# Replace impossible spikes (e.g. the source's 123k reading on 2026-08-06 16:00)
# with values interpolated from the surrounding 15-minute intervals.
OUTLIER_THRESHOLD = 5000
for col in ["Redemption Count", "Sales Count"]:
    outliers = df[col] > OUTLIER_THRESHOLD
    if outliers.any():
        for _, row in df.loc[outliers].iterrows():
            print(f"Outlier in {col} at {row['Timestamp']}: {row[col]} -> interpolating from neighbours")
        df[col] = (
            df[col].mask(outliers)
            .interpolate(limit_direction="both")
            .round()
            .astype(int)
        )

# Daily totals
df["date"] = df["Timestamp"].dt.date
daily = df.groupby("date").agg(
    redemptions=("Redemption Count", "sum"),
).reset_index()

daily_path = outputs / "daily_totals.csv"
daily.to_csv(daily_path, index=False)
print(f"Saved {len(daily)} daily records to {daily_path}")
print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

# Hourly totals
df["hour"] = df["Timestamp"].dt.floor("h")
hourly = df.groupby("hour").agg(
    redemptions=("Redemption Count", "sum"),
).reset_index()

hourly_path = outputs / "hourly_totals.csv"
hourly.to_csv(hourly_path, index=False)
print(f"Saved {len(hourly)} hourly records to {hourly_path}")
