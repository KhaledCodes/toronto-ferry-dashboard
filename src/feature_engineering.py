"""
Feature engineering for Toronto Ferry ridership forecasting.
Creates lag features, rolling statistics, and prepares data for ML.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

from .config import (
    LAG_PERIODS, ROLLING_WINDOWS, WEATHER_FEATURES,
    TEMPORAL_FEATURES, CALENDAR_EVENT_FEATURES, CATEGORICAL_FEATURES,
    TARGET_REDEMPTION, TARGET_SALES, TOURISM_FEATURES
)
from .tourism_features import add_tourism_features


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-minute data to hourly for better ML performance.

    Args:
        df: DataFrame with 15-minute granularity

    Returns:
        DataFrame aggregated to hourly level
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Use timestamp_hour for grouping
    if 'timestamp_hour' in df.columns:
        df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
        group_col = 'timestamp_hour'
    else:
        df['timestamp_hour'] = df['timestamp'].dt.floor('h')
        group_col = 'timestamp_hour'

    # Custom sum function that returns NaN if all values are NaN
    def nan_sum(x):
        if x.isna().all():
            return np.nan
        return x.sum()

    # Aggregation rules
    agg_dict = {
        # Sum the counts (4 x 15-min periods per hour), preserve NaN for future
        TARGET_REDEMPTION: nan_sum,
        TARGET_SALES: nan_sum,
    }

    # For weather features, take the mean (they're already hourly aligned anyway)
    for col in WEATHER_FEATURES:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # For categorical/temporal features, take the first value (same within hour)
    other_cols = TEMPORAL_FEATURES + CALENDAR_EVENT_FEATURES + ['season', 'holiday_name']
    for col in other_cols:
        if col in df.columns:
            agg_dict[col] = 'first'

    # Group and aggregate
    hourly_df = df.groupby(group_col).agg(agg_dict).reset_index()
    hourly_df = hourly_df.rename(columns={group_col: 'timestamp'})

    # Sort by timestamp ascending (important for lag features)
    hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)

    return hourly_df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lagged versions of target variable.

    Args:
        df: DataFrame sorted by timestamp ascending
        target_col: Column to create lags for
        lags: List of lag periods in hours (default: [1, 6, 24, 168])

    Returns:
        DataFrame with new lag columns
    """
    if lags is None:
        lags = LAG_PERIODS

    df = df.copy()

    for lag in lags:
        col_name = f'{target_col}_lag_{lag}h'
        df[col_name] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Create rolling statistics (mean, std).

    Args:
        df: DataFrame sorted by timestamp ascending
        target_col: Column to create rolling stats for
        windows: List of window sizes in hours (default: [24, 168])

    Returns:
        DataFrame with rolling_mean_*, rolling_std_* columns
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()

    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}h'] = (
            df[target_col]
            .shift(1)  # Don't include current value (data leakage prevention)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        # Rolling std
        df[f'{target_col}_rolling_std_{window}h'] = (
            df[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)  # First values have no std
        )

    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hour and day_of_week to cyclical sin/cos features.
    Helps model learn that hour 23 is close to hour 0.

    Args:
        df: DataFrame with hour_of_day and day_of_week columns

    Returns:
        DataFrame with cyclical features added
    """
    df = df.copy()

    # Hour cyclical encoding (0-23 cycle)
    if 'hour_of_day' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # Day of week cyclical encoding (0-6 cycle)
    if 'day_of_week' in df.columns:
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month cyclical encoding (1-12 cycle)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = TARGET_REDEMPTION,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_cyclical: bool = True,
    include_tourism: bool = True
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw features DataFrame (can be 15-min or hourly)
        target_col: Target column for lag/rolling features
        include_lags: Whether to add lag features
        include_rolling: Whether to add rolling statistics
        include_cyclical: Whether to add cyclical encodings
        include_tourism: Whether to add tourism/World Cup features

    Returns:
        Feature-engineered DataFrame ready for ML
    """
    # Check if we need to aggregate to hourly
    df = df.copy()

    # Ensure sorted by timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    elif 'timestamp_hour' in df.columns:
        df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
        df = df.sort_values('timestamp_hour').reset_index(drop=True)

    # Check granularity and aggregate if needed
    if 'timestamp' in df.columns:
        time_diff = df['timestamp'].diff().median()
        if time_diff and time_diff < pd.Timedelta(hours=1):
            df = aggregate_to_hourly(df)

    # Add lag features
    if include_lags and target_col in df.columns:
        df = create_lag_features(df, target_col)

    # Add rolling features
    if include_rolling and target_col in df.columns:
        df = create_rolling_features(df, target_col)

    # Add cyclical features
    if include_cyclical:
        df = create_cyclical_features(df)

    # Add tourism and World Cup features
    if include_tourism:
        # Check if tourism features already exist
        if 'tourism_index' not in df.columns:
            df = add_tourism_features(df)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = TARGET_REDEMPTION) -> List[str]:
    """
    Get list of feature columns to use for training.
    Excludes timestamp, target, and other non-feature columns.

    Args:
        df: DataFrame with all columns
        target_col: Target column to exclude

    Returns:
        List of feature column names
    """
    exclude_cols = {
        'timestamp', 'timestamp_hour', 'time',
        TARGET_REDEMPTION, TARGET_SALES,
        'holiday_name', 'season'  # These are string columns, use encoded versions
    }

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def split_train_test_by_time(
    df: pd.DataFrame,
    test_months: int = 1,
    val_months: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically for proper time series validation.

    Args:
        df: DataFrame with timestamp column, sorted ascending
        test_months: Number of months for test set
        val_months: Number of months for validation set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()

    # Get timestamp column
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_hour'
    df[ts_col] = pd.to_datetime(df[ts_col])

    max_date = df[ts_col].max()
    test_start = max_date - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    train_df = df[df[ts_col] < val_start].copy()
    val_df = df[(df[ts_col] >= val_start) & (df[ts_col] < test_start)].copy()
    test_df = df[df[ts_col] >= test_start].copy()

    return train_df, val_df, test_df


def encode_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode season column as numeric if present.
    """
    df = df.copy()
    if 'season' in df.columns:
        season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
        df['season_encoded'] = df['season'].map(season_map).fillna(0).astype(int)
    return df
