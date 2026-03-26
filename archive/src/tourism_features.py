"""
Tourism and special event features for Toronto Ferry ridership forecasting.
Includes historical tourism data and 2026 FIFA World Cup event features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# Toronto monthly tourism index based on Destination Toronto data
# Normalized to average = 100 (based on 2024 data: 9M total overnight visitors)
# Peak: August (1.01M), July (965K); Low: January-February
MONTHLY_TOURISM_INDEX = {
    1: 55,    # January - winter low
    2: 58,    # February
    3: 72,    # March - spring break bump
    4: 85,    # April
    5: 105,   # May - shoulder season starts
    6: 125,   # June - summer begins
    7: 140,   # July - peak summer
    8: 145,   # August - peak
    9: 115,   # September - Labour Day
    10: 95,   # October - fall
    11: 75,   # November
    12: 70,   # December - holiday dip then bump
}

# Year-over-year tourism growth factors (relative to 2019 baseline)
# Based on Statistics Canada and Destination Toronto recovery data
YEARLY_TOURISM_FACTOR = {
    2018: 0.98,
    2019: 1.00,   # Baseline
    2020: 0.25,   # COVID crash
    2021: 0.35,   # Partial recovery
    2022: 0.70,   # Strong recovery
    2023: 0.88,   # Near full recovery
    2024: 0.95,   # Full recovery
    2025: 1.00,   # Back to baseline
    2026: 1.15,   # World Cup boost (projected 300K+ additional visitors)
    2027: 1.02,   # Post-WC normalization
}

# 2026 FIFA World Cup Toronto match schedule
# Toronto hosts 6 matches at BMO Field
# Tournament: June 11 - July 19, 2026
WORLD_CUP_2026 = {
    'tournament_start': datetime(2026, 6, 11),
    'tournament_end': datetime(2026, 7, 19),
    # Toronto match dates (BMO Field - estimated schedule)
    # Canada opening match + 5 other group/knockout matches
    'toronto_match_dates': [
        datetime(2026, 6, 12),  # Canada opening match (estimated)
        datetime(2026, 6, 16),
        datetime(2026, 6, 20),
        datetime(2026, 6, 24),
        datetime(2026, 6, 28),
        datetime(2026, 7, 2),   # Round of 32 (if hosting)
    ],
    # Expected visitor boost during tournament (multiplier)
    'base_boost': 1.3,          # 30% boost during tournament period
    'match_day_boost': 1.8,     # 80% boost on Toronto match days
    'opening_match_boost': 2.5, # 150% boost for Canada's opening
}


def get_tourism_index(date: datetime) -> float:
    """
    Get tourism index for a given date.
    Combines monthly seasonality with yearly growth trends.

    Args:
        date: Date to get index for

    Returns:
        Tourism index value (100 = baseline average)
    """
    month = date.month
    year = date.year

    # Get monthly base
    monthly_idx = MONTHLY_TOURISM_INDEX.get(month, 100)

    # Apply yearly factor
    yearly_factor = YEARLY_TOURISM_FACTOR.get(year, 1.0)

    return monthly_idx * yearly_factor


def get_world_cup_features(date: datetime) -> dict:
    """
    Get World Cup-related features for a given date.

    Args:
        date: Date to get features for

    Returns:
        Dictionary of World Cup features
    """
    wc = WORLD_CUP_2026

    # Default values
    features = {
        'is_world_cup_period': 0,
        'is_world_cup_match_day': 0,
        'is_canada_opening': 0,
        'days_to_world_cup': 0,
        'days_into_world_cup': 0,
        'world_cup_boost': 1.0,
    }

    # Check if date is a datetime or just date
    if hasattr(date, 'date'):
        check_date = date.date()
    else:
        check_date = date

    tournament_start = wc['tournament_start'].date()
    tournament_end = wc['tournament_end'].date()

    # Days to/from World Cup
    days_diff = (tournament_start - check_date).days
    features['days_to_world_cup'] = max(0, days_diff)

    # During tournament period
    if tournament_start <= check_date <= tournament_end:
        features['is_world_cup_period'] = 1
        features['days_into_world_cup'] = (check_date - tournament_start).days
        features['world_cup_boost'] = wc['base_boost']

        # Check for Toronto match day
        for match_date in wc['toronto_match_dates']:
            if check_date == match_date.date():
                features['is_world_cup_match_day'] = 1
                features['world_cup_boost'] = wc['match_day_boost']

                # Canada opening match (first in list)
                if match_date == wc['toronto_match_dates'][0]:
                    features['is_canada_opening'] = 1
                    features['world_cup_boost'] = wc['opening_match_boost']
                break

    # Pre-tournament buildup (2 weeks before)
    elif 0 < days_diff <= 14:
        features['world_cup_boost'] = 1.0 + (14 - days_diff) * 0.02  # Gradual buildup

    return features


def add_tourism_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Add tourism and World Cup features to a DataFrame.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with tourism features added
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    elif 'timestamp_hour' in df.columns:
        timestamp_col = 'timestamp_hour'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    else:
        raise ValueError("No timestamp column found")

    # Add tourism index
    df['tourism_index'] = df[timestamp_col].apply(get_tourism_index)

    # Add World Cup features
    wc_features = df[timestamp_col].apply(get_world_cup_features)
    wc_df = pd.DataFrame(wc_features.tolist(), index=df.index)

    # Merge World Cup features
    for col in wc_df.columns:
        df[col] = wc_df[col]

    return df


def get_special_events_calendar() -> pd.DataFrame:
    """
    Get a calendar of major Toronto events that affect ferry ridership.
    These supplement the existing holiday calendar.

    Returns:
        DataFrame with event dates and expected impact multipliers
    """
    events = [
        # 2026 World Cup (already handled separately)

        # Annual recurring events (dates approximate)
        {'event': 'CNE', 'month': 8, 'day_start': 16, 'duration': 18, 'boost': 1.2},
        {'event': 'Toronto Caribbean Carnival', 'month': 8, 'day_start': 1, 'duration': 3, 'boost': 1.15},
        {'event': 'TIFF', 'month': 9, 'day_start': 5, 'duration': 11, 'boost': 1.1},
        {'event': 'Pride Month Peak', 'month': 6, 'day_start': 25, 'duration': 5, 'boost': 1.25},
        {'event': 'Canada Day', 'month': 7, 'day_start': 1, 'duration': 1, 'boost': 1.5},
        {'event': 'Victoria Day Weekend', 'month': 5, 'day_start': 18, 'duration': 3, 'boost': 1.3},
        {'event': 'Civic Holiday Weekend', 'month': 8, 'day_start': 1, 'duration': 3, 'boost': 1.25},
    ]

    return pd.DataFrame(events)


def create_tourism_projection(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'h'
) -> pd.DataFrame:
    """
    Create a tourism projection DataFrame for a date range.
    Useful for forecasting periods.

    Args:
        start_date: Start of projection period
        end_date: End of projection period
        freq: Frequency ('H' for hourly, 'D' for daily)

    Returns:
        DataFrame with timestamp and all tourism features
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    df = pd.DataFrame({'timestamp': dates})

    # Add all tourism features
    df = add_tourism_features(df, 'timestamp')

    return df


# List of tourism feature columns for use in model training
TOURISM_FEATURE_COLUMNS = [
    'tourism_index',
    'is_world_cup_period',
    'is_world_cup_match_day',
    'is_canada_opening',
    'days_to_world_cup',
    'days_into_world_cup',
    'world_cup_boost',
]
