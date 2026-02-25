"""
Daily update script for Toronto Ferry Dashboard.
This script:
1. Fetches latest ferry ticket data from Toronto Open Data
2. Fetches weather data and forecasts
3. Updates the historical data CSV
4. Generates new 7-day and long-term forecasts

Usage: python scripts/daily_update.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np
import requests
import holidays
import meteostat as ms
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_inference import (
    generate_forecasts, create_daily_summary,
    generate_long_term_forecasts
)
from src.config import OUTPUTS_DIR, HOURLY_DATA_CSV

# Constants
TORONTO_STATION_ID = 71508  # Meteostat station ID for Toronto
TORONTO_LAT = 43.61
TORONTO_LON = -79.42


def fetch_ferry_data():
    """Fetch latest ferry ticket data from Toronto Open Data."""
    print("Fetching ferry data from Toronto Open Data...")

    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = base_url + "/api/3/action/package_show"
    params = {"id": "toronto-island-ferry-ticket-counts"}

    try:
        package = requests.get(url, params=params, timeout=30).json()

        for resource in package["result"]["resources"]:
            if resource["datastore_active"]:
                dump_url = base_url + "/datastore/dump/" + resource["id"]
                response = requests.get(dump_url, timeout=60)
                df = pd.read_csv(StringIO(response.text))

                # Clean up columns
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                if '_id' in df.columns:
                    df = df.drop(['_id'], axis=1)

                df = df.rename(columns={
                    'Timestamp': 'timestamp',
                    'Redemption Count': 'redemption_count',
                    'Sales Count': 'sales_count'
                })

                print(f"  Fetched {len(df)} records from API")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                return df

    except Exception as e:
        print(f"Error fetching ferry data: {e}")
        return None


def fetch_weather_data(start_date, end_date):
    """Fetch weather data from Meteostat."""
    print(f"Fetching weather data from {start_date} to {end_date}...")

    try:
        data = ms.hourly(str(TORONTO_STATION_ID), start_date, end_date, timezone='America/Toronto')
        df = data.fetch()
        df = df.reset_index()
        print(f"  Fetched {len(df)} hourly weather records")
        return df
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def fetch_weather_forecast():
    """Fetch weather forecast from Open-Meteo."""
    print("Fetching weather forecast from Open-Meteo...")

    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": TORONTO_LAT,
            "longitude": TORONTO_LON,
            "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m"],
            "timezone": "America/Toronto",
            "forecast_days": 7
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        hourly = response.Hourly()
        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ).tz_convert('America/Toronto'),
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "prcp": hourly.Variables(1).ValuesAsNumpy(),
            "rhum": hourly.Variables(2).ValuesAsNumpy(),
            "wspd": hourly.Variables(3).ValuesAsNumpy(),
        }

        df = pd.DataFrame(data=hourly_data)
        print(f"  Fetched {len(df)} hours of forecast")
        return df

    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        return None


def add_time_features(df):
    """Add time-based features to the dataframe."""
    df = df.copy()

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

    # Canadian holidays
    canadian_holidays = holidays.CA(prov='ON', years=range(2015, 2035))
    df['is_holiday'] = df['timestamp'].dt.date.isin(canadian_holidays).astype(int)

    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['month'].apply(get_season)

    # School breaks
    def is_school_break(row):
        month = row['month']
        day = row['day_of_month']
        if month == 12 and day >= 23:
            return 1
        if month == 1 and day <= 7:
            return 1
        if month == 3 and 11 <= day <= 15:
            return 1
        if month in [7, 8]:
            return 1
        return 0

    df['is_school_break'] = df.apply(is_school_break, axis=1)

    # Days since/until weekend
    def days_since_weekend(dow):
        return dow + 1 if dow < 5 else 0

    def days_until_weekend(dow):
        return abs(dow - 5) if dow < 5 else 0

    df['days_since_weekend'] = df['day_of_week'].apply(days_since_weekend)
    df['days_until_weekend'] = df['day_of_week'].apply(days_until_weekend)

    # COVID and flooding flags
    df['is_covid_lockdown'] = ((df['timestamp'] >= '2020-03-17') &
                               (df['timestamp'] <= '2021-12-31')).astype(int)
    df['is_flooding'] = ((df['timestamp'] >= '2017-05-04') &
                         (df['timestamp'] <= '2017-07-31')).astype(int)

    return df


def update_hourly_data():
    """Update the hourly_data.csv with latest ferry and weather data."""
    print("\n" + "="*60)
    print(" Updating Historical Data")
    print("="*60)

    hourly_path = OUTPUTS_DIR / "hourly_data.csv"

    # Load existing data
    if hourly_path.exists():
        existing_df = pd.read_csv(hourly_path)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        last_timestamp = existing_df['timestamp'].max()
        print(f"Existing data up to: {last_timestamp}")
    else:
        existing_df = None
        last_timestamp = datetime(2015, 5, 1)
        print("No existing data found, starting from scratch")

    # Fetch new ferry data
    ferry_df = fetch_ferry_data()
    if ferry_df is None:
        print("Failed to fetch ferry data, skipping update")
        return False

    # Fetch weather data (from last timestamp to now + 7 days for forecasts)
    weather_start = last_timestamp - timedelta(days=1)  # Overlap for safety
    weather_end = datetime.now() + timedelta(days=1)
    weather_df = fetch_weather_data(weather_start, weather_end)

    if weather_df is None:
        print("Failed to fetch weather data, skipping update")
        return False

    # Process ferry data - aggregate to hourly
    ferry_df['timestamp_hour'] = ferry_df['timestamp'].dt.floor('h')
    hourly_ferry = ferry_df.groupby('timestamp_hour').agg({
        'redemption_count': 'sum',
        'sales_count': 'sum'
    }).reset_index()
    hourly_ferry = hourly_ferry.rename(columns={'timestamp_hour': 'timestamp'})

    # Process weather data
    weather_df = weather_df.rename(columns={'time': 'timestamp'})
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp']).dt.tz_localize(None)

    # Fetch 7-day weather forecast from Open-Meteo
    forecast_df = fetch_weather_forecast()
    if forecast_df is not None:
        forecast_df = forecast_df.rename(columns={'time': 'timestamp'})
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp']).dt.tz_localize(None)
        # Combine historical weather with forecast, preferring historical for overlapping hours
        weather_combined = pd.concat([weather_df, forecast_df], ignore_index=True)
        weather_combined = weather_combined.drop_duplicates(subset=['timestamp'], keep='first')
        weather_combined = weather_combined.sort_values('timestamp').reset_index(drop=True)
        print(f"  Combined weather: {len(weather_combined)} hours (historical + 7-day forecast)")
    else:
        weather_combined = weather_df
        print("  Using historical weather only (forecast fetch failed)")

    # Merge ferry and weather
    weather_cols = [c for c in ['timestamp', 'temp', 'prcp', 'rhum', 'wspd', 'wdir', 'pres'] if c in weather_combined.columns]
    merged_df = pd.merge(
        hourly_ferry,
        weather_combined[weather_cols],
        on='timestamp',
        how='outer'
    )

    # Fill missing weather values with interpolation
    for col in ['temp', 'prcp', 'rhum', 'wspd', 'wdir', 'pres']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].interpolate()

    # Fill missing ferry counts with 0 for past dates
    now = datetime.now()
    merged_df.loc[merged_df['timestamp'] < now, 'redemption_count'] = \
        merged_df.loc[merged_df['timestamp'] < now, 'redemption_count'].fillna(0)
    merged_df.loc[merged_df['timestamp'] < now, 'sales_count'] = \
        merged_df.loc[merged_df['timestamp'] < now, 'sales_count'].fillna(0)

    # Add time features
    merged_df = add_time_features(merged_df)

    # Combine with existing data
    if existing_df is not None:
        # Only keep new records
        new_records = merged_df[merged_df['timestamp'] > last_timestamp]
        if len(new_records) > 0:
            combined_df = pd.concat([existing_df, new_records], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        else:
            combined_df = existing_df
            print("No new records to add")
    else:
        combined_df = merged_df

    # Save updated data
    combined_df.to_csv(hourly_path, index=False)
    print(f"\nUpdated hourly_data.csv: {len(combined_df)} total records")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

    return True


def generate_all_forecasts():
    """Generate both 7-day and long-term forecasts."""
    print("\n" + "="*60)
    print(" Generating Forecasts")
    print("="*60)

    # Generate 7-day forecasts
    print("\n--- 7-Day Forecast ---")
    try:
        forecasts = generate_forecasts(features_path=str(HOURLY_DATA_CSV))

        if len(forecasts) > 0:
            daily = create_daily_summary(forecasts)
            daily_path = OUTPUTS_DIR / "daily_forecasts.csv"
            daily.to_csv(daily_path, index=False)
            print(f"7-day forecast saved: {len(forecasts)} hourly predictions")

            # Print summary
            print(f"\n{'Date':<12} {'Redemptions':>12} {'Sales':>10}")
            print("-"*40)
            for _, row in daily.iterrows():
                print(f"{str(row['date']):<12} {row['total_redemptions']:>12,} {row['total_sales']:>10,}")
        else:
            print("No 7-day forecasts generated")
    except Exception as e:
        print(f"Error generating 7-day forecasts: {e}")

    # Generate long-term forecasts
    print("\n--- Long-Term (365-Day) Forecast ---")
    try:
        long_term = generate_long_term_forecasts(features_path=str(HOURLY_DATA_CSV))

        if len(long_term) > 0:
            print(f"Long-term forecast saved: {len(long_term)} hourly predictions")

            # Print monthly summary
            monthly_path = OUTPUTS_DIR / "long_term_monthly_forecasts.csv"
            if monthly_path.exists():
                monthly = pd.read_csv(monthly_path)
                print(f"\n{'Month':<10} {'Redemptions':>12}")
                print("-"*25)
                for _, row in monthly.head(6).iterrows():
                    print(f"{row['year_month']:<10} {row['total_redemptions']:>12,}")
                print("...")
        else:
            print("No long-term forecasts generated (no-weather models may not be trained)")
    except Exception as e:
        print(f"Error generating long-term forecasts: {e}")


def main():
    print("="*60)
    print(" Toronto Ferry Dashboard - Daily Update")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Update historical data
    success = update_hourly_data()

    if success:
        # Generate forecasts
        generate_all_forecasts()

    print("\n" + "="*60)
    print(" Update Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
