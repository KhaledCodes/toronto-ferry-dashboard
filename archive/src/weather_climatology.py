"""
Weather Climatology Module

Computes historical weather averages by day-of-year and hour-of-day
for use in long-term forecasting beyond the 7-day weather forecast window.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from .config import WEATHER_FEATURES, HOURLY_DATA_CSV, OUTPUTS_DIR


def compute_weather_climatology(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical weather averages by day-of-year and hour-of-day.

    Args:
        hourly_df: DataFrame with timestamp and weather columns

    Returns:
        DataFrame with climatology statistics for each day-of-year/hour combination
    """
    df = hourly_df.copy()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['hour_of_day'] = df['timestamp'].dt.hour

    # Group by day-of-year and hour-of-day
    grouped = df.groupby(['day_of_year', 'hour_of_day'])

    # Compute statistics for each weather feature
    climatology_records = []

    for (doy, hour), group in grouped:
        record = {
            'day_of_year': doy,
            'hour_of_day': hour,
        }

        for feature in WEATHER_FEATURES:
            if feature in group.columns:
                values = group[feature].dropna()
                if len(values) > 0:
                    record[f'{feature}_mean'] = values.mean()
                    record[f'{feature}_std'] = values.std() if len(values) > 1 else 0
                    record[f'{feature}_p10'] = values.quantile(0.1)
                    record[f'{feature}_p25'] = values.quantile(0.25)
                    record[f'{feature}_p50'] = values.quantile(0.5)
                    record[f'{feature}_p75'] = values.quantile(0.75)
                    record[f'{feature}_p90'] = values.quantile(0.9)
                else:
                    record[f'{feature}_mean'] = np.nan
                    record[f'{feature}_std'] = np.nan
                    record[f'{feature}_p10'] = np.nan
                    record[f'{feature}_p25'] = np.nan
                    record[f'{feature}_p50'] = np.nan
                    record[f'{feature}_p75'] = np.nan
                    record[f'{feature}_p90'] = np.nan

        climatology_records.append(record)

    climatology_df = pd.DataFrame(climatology_records)

    # Handle leap year: day 366 (Feb 29) - copy from day 60 (Feb 29 equivalent)
    # If day 366 doesn't exist, create it from day 60
    if 366 not in climatology_df['day_of_year'].values:
        day_60_data = climatology_df[climatology_df['day_of_year'] == 60].copy()
        if len(day_60_data) > 0:
            day_60_data['day_of_year'] = 366
            climatology_df = pd.concat([climatology_df, day_60_data], ignore_index=True)

    # Fill any remaining NaN values using forward/backward fill
    climatology_df = climatology_df.sort_values(['day_of_year', 'hour_of_day'])
    climatology_df = climatology_df.fillna(method='ffill').fillna(method='bfill')

    return climatology_df


def get_climatology_for_datetime(
    climatology_df: pd.DataFrame,
    target_datetime: datetime
) -> dict:
    """
    Get climatology values for a specific datetime.

    Args:
        climatology_df: Pre-computed climatology DataFrame
        target_datetime: The datetime to get climatology for

    Returns:
        Dictionary with weather feature means for that day/hour
    """
    doy = target_datetime.timetuple().tm_yday
    hour = target_datetime.hour

    # Handle leap year edge case
    if doy > 366:
        doy = 366

    row = climatology_df[
        (climatology_df['day_of_year'] == doy) &
        (climatology_df['hour_of_day'] == hour)
    ]

    if len(row) == 0:
        # Fallback: use nearest day
        row = climatology_df[climatology_df['hour_of_day'] == hour].iloc[0:1]

    result = {}
    for feature in WEATHER_FEATURES:
        mean_col = f'{feature}_mean'
        if mean_col in row.columns:
            result[feature] = row[mean_col].values[0]
        else:
            result[feature] = np.nan

    return result


def generate_climatology_features(
    start_date: datetime,
    end_date: datetime,
    climatology_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate weather features for a date range using climatology data.

    Args:
        start_date: Start of the forecast period
        end_date: End of the forecast period
        climatology_df: Pre-computed climatology DataFrame

    Returns:
        DataFrame with hourly timestamps and climatology-based weather features
    """
    # Generate hourly timestamps
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')

    records = []
    for dt in date_range:
        weather = get_climatology_for_datetime(climatology_df, dt)
        weather['timestamp'] = dt
        records.append(weather)

    df = pd.DataFrame(records)
    df = df[['timestamp'] + WEATHER_FEATURES]

    return df


def load_or_compute_climatology(
    hourly_data_path: Optional[Path] = None,
    climatology_path: Optional[Path] = None,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load existing climatology or compute it from historical data.

    Args:
        hourly_data_path: Path to hourly historical data
        climatology_path: Path to save/load climatology CSV
        force_recompute: If True, always recompute even if file exists

    Returns:
        Climatology DataFrame
    """
    if hourly_data_path is None:
        hourly_data_path = HOURLY_DATA_CSV
    if climatology_path is None:
        climatology_path = OUTPUTS_DIR / "weather_climatology.csv"

    if climatology_path.exists() and not force_recompute:
        return pd.read_csv(climatology_path)

    # Load historical data and compute climatology
    print("Computing weather climatology from historical data...")
    hourly_df = pd.read_csv(hourly_data_path)
    climatology_df = compute_weather_climatology(hourly_df)

    # Save for future use
    climatology_df.to_csv(climatology_path, index=False)
    print(f"Saved climatology to {climatology_path}")

    return climatology_df


def get_climatology_uncertainty(
    climatology_df: pd.DataFrame,
    target_datetime: datetime,
    feature: str = 'temp'
) -> tuple:
    """
    Get uncertainty bounds for a weather feature at a specific datetime.

    Args:
        climatology_df: Pre-computed climatology DataFrame
        target_datetime: The datetime to get uncertainty for
        feature: Weather feature name

    Returns:
        Tuple of (p10, p50, p90) percentile values
    """
    doy = target_datetime.timetuple().tm_yday
    hour = target_datetime.hour

    row = climatology_df[
        (climatology_df['day_of_year'] == doy) &
        (climatology_df['hour_of_day'] == hour)
    ]

    if len(row) == 0:
        return (np.nan, np.nan, np.nan)

    p10 = row[f'{feature}_p10'].values[0]
    p50 = row[f'{feature}_p50'].values[0]
    p90 = row[f'{feature}_p90'].values[0]

    return (p10, p50, p90)
