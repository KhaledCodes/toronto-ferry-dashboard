"""
Aggregate 15-minute ferry ticket data to daily and hourly totals for the dashboard.
"""
import pandas as pd
from pathlib import Path

outputs = Path(__file__).parent.parent / "outputs"

df = pd.read_csv(outputs / "ferry_ticket_counts.csv", parse_dates=["Timestamp"])

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
