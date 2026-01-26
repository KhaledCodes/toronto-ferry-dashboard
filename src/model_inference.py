"""
Model inference for Toronto Ferry ridership forecasting.
Generates predictions for future dates using weather forecasts.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from .config import (
    FEATURES_CSV, FORECASTS_CSV, OUTPUTS_DIR,
    REDEMPTION_MODEL_PATH, SALES_MODEL_PATH,
    REDEMPTION_NO_WEATHER_MODEL_PATH, SALES_NO_WEATHER_MODEL_PATH,
    TARGET_REDEMPTION, TARGET_SALES, FORECAST_HORIZON_DAYS,
    LONG_TERM_HORIZON_DAYS, LONG_TERM_FORECASTS_CSV,
    LONG_TERM_DAILY_CSV, LONG_TERM_MONTHLY_CSV,
    BLEND_TRANSITION_START, BLEND_TRANSITION_END, WEATHER_CLIMATOLOGY_CSV,
)
from .feature_engineering import prepare_features, encode_season, aggregate_to_hourly
from .model_training import FerryRidershipModel
from .weather_climatology import load_or_compute_climatology, generate_climatology_features
from .tourism_features import add_tourism_features


def load_models() -> Tuple[FerryRidershipModel, FerryRidershipModel]:
    """
    Load both trained weather models.

    Returns:
        Tuple of (redemption_model, sales_model)
    """
    redemption_model = FerryRidershipModel.load(REDEMPTION_MODEL_PATH)
    sales_model = FerryRidershipModel.load(SALES_MODEL_PATH)
    return redemption_model, sales_model


def load_no_weather_models() -> Tuple[FerryRidershipModel, FerryRidershipModel]:
    """
    Load both trained no-weather models for long-term forecasting.

    Returns:
        Tuple of (redemption_model_no_weather, sales_model_no_weather)
    """
    redemption_model = FerryRidershipModel.load(REDEMPTION_NO_WEATHER_MODEL_PATH)
    sales_model = FerryRidershipModel.load(SALES_NO_WEATHER_MODEL_PATH)
    return redemption_model, sales_model


def compute_blend_weight(forecast_day: int) -> float:
    """
    Compute blending weight for combining weather and no-weather model predictions.

    Args:
        forecast_day: Number of days into the future (1 = tomorrow)

    Returns:
        Weight for weather model (1 - weight = no-weather model weight)
        - Days 1-5: 1.0 (100% weather model)
        - Days 5-8: Linear decay from 1.0 to 0.0
        - Days 8+: 0.0 (100% no-weather model)
    """
    if forecast_day <= BLEND_TRANSITION_START:
        return 1.0
    elif forecast_day >= BLEND_TRANSITION_END:
        return 0.0
    else:
        # Linear interpolation
        return 1.0 - (forecast_day - BLEND_TRANSITION_START) / (BLEND_TRANSITION_END - BLEND_TRANSITION_START)


def blend_predictions(
    weather_pred: float,
    no_weather_pred: float,
    forecast_day: int
) -> float:
    """
    Blend predictions from weather and no-weather models.

    Args:
        weather_pred: Prediction from weather model
        no_weather_pred: Prediction from no-weather model
        forecast_day: Number of days into the future

    Returns:
        Blended prediction
    """
    weight = compute_blend_weight(forecast_day)
    return weight * weather_pred + (1 - weight) * no_weather_pred


def compute_prediction_interval(
    point_pred: float,
    forecast_day: int,
    base_std: float = 50.0
) -> Tuple[float, float]:
    """
    Compute prediction interval that widens with forecast horizon.

    Args:
        point_pred: Point prediction value
        forecast_day: Number of days into the future
        base_std: Base standard deviation for uncertainty

    Returns:
        Tuple of (lower_bound, upper_bound) for ~80% confidence interval
    """
    # Uncertainty grows with sqrt of forecast horizon
    uncertainty_factor = 1.0 + 0.15 * np.sqrt(forecast_day)
    margin = 1.28 * base_std * uncertainty_factor  # ~80% CI

    lower = max(0, point_pred - margin)
    upper = point_pred + margin
    return lower, upper


def prepare_forecast_data(
    df: pd.DataFrame,
    historical_df: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    """
    Prepare forecast data with lag features computed from historical data.

    Args:
        df: DataFrame with future rows (target is NaN)
        historical_df: DataFrame with historical data (target has values)
        target_col: Target column name

    Returns:
        DataFrame ready for prediction
    """
    from .config import LAG_PERIODS, ROLLING_WINDOWS

    historical_df = historical_df.copy()
    df = df.copy()

    # Sort historical data
    historical_df = historical_df.sort_values('timestamp').reset_index(drop=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Ensure both have the target column
    if target_col not in df.columns:
        df[target_col] = np.nan

    # Get last N hours of historical data for computing lag features
    # We need at least 168 hours (1 week) for the longest lag
    max_lag = max(LAG_PERIODS)  # 168 hours
    last_historical = historical_df.tail(max_lag + 1).copy()

    # Concatenate last historical + future for proper lag computation
    combined = pd.concat([last_historical, df], ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Compute lag features manually using only historical values
    for lag in LAG_PERIODS:
        col_name = f'{target_col}_lag_{lag}h'
        combined[col_name] = combined[target_col].shift(lag)

    # Compute rolling features using only historical values
    for window in ROLLING_WINDOWS:
        combined[f'{target_col}_rolling_mean_{window}h'] = (
            combined[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        combined[f'{target_col}_rolling_std_{window}h'] = (
            combined[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )

    # For future rows, use lag_168h (same hour 1 week ago) as proxy for ALL lags
    # This is the only lag that can be correctly computed for multi-day forecasts
    # since our forecast horizon is <= 7 days
    future_mask = combined[target_col].isna()

    if f'{target_col}_lag_168h' in combined.columns:
        # Use 1-week-ago values as proxy for all lag features
        lag_168h_values = combined.loc[future_mask, f'{target_col}_lag_168h']

        # Fill short-term lags with 1-week-ago value (same hour, same day of week)
        combined.loc[future_mask, f'{target_col}_lag_1h'] = lag_168h_values.values
        combined.loc[future_mask, f'{target_col}_lag_6h'] = lag_168h_values.values
        combined.loc[future_mask, f'{target_col}_lag_24h'] = lag_168h_values.values

        # For rolling means, use the 168h values
        if f'{target_col}_rolling_mean_168h' in combined.columns:
            combined.loc[future_mask, f'{target_col}_rolling_mean_24h'] = \
                combined.loc[future_mask, f'{target_col}_rolling_mean_168h'].values

    # Forward-fill any remaining NaNs
    lag_cols = [c for c in combined.columns if 'lag' in c or 'rolling' in c]
    combined[lag_cols] = combined[lag_cols].ffill()

    # Apply other feature engineering (cyclical, season encoding)
    combined = encode_season(combined)

    # Add cyclical features
    if 'hour_of_day' in combined.columns:
        combined['hour_sin'] = np.sin(2 * np.pi * combined['hour_of_day'] / 24)
        combined['hour_cos'] = np.cos(2 * np.pi * combined['hour_of_day'] / 24)
    if 'day_of_week' in combined.columns:
        combined['dow_sin'] = np.sin(2 * np.pi * combined['day_of_week'] / 7)
        combined['dow_cos'] = np.cos(2 * np.pi * combined['day_of_week'] / 7)
    if 'month' in combined.columns:
        combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12)
        combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12)

    # Add tourism and World Cup features
    if 'tourism_index' not in combined.columns:
        combined = add_tourism_features(combined)

    # Extract only the future rows (those with NaN target)
    future_mask = combined[target_col].isna()
    forecast_df = combined[future_mask].copy()

    return forecast_df


def generate_forecasts(
    features_path: str = None,
    output_path: str = None,
    horizon_days: int = FORECAST_HORIZON_DAYS
) -> pd.DataFrame:
    """
    Generate forecasts for future dates.

    Args:
        features_path: Path to features.csv (default: FEATURES_CSV)
        output_path: Path to save forecasts (default: FORECASTS_CSV)
        horizon_days: Days to forecast (limited by weather forecast availability)

    Returns:
        DataFrame with predictions
    """
    if features_path is None:
        features_path = FEATURES_CSV
    if output_path is None:
        output_path = FORECASTS_CSV

    print(f"Loading data from {features_path}...")

    # Load full dataset
    df = pd.read_csv(features_path)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'timestamp_hour' in df.columns:
        df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])

    # Aggregate to hourly
    print("Aggregating to hourly...")
    hourly_df = aggregate_to_hourly(df)

    # Split into historical (has target values) and future (target is NaN)
    historical_mask = hourly_df[TARGET_REDEMPTION].notna()
    historical_df = hourly_df[historical_mask].copy()
    future_df = hourly_df[~historical_mask].copy()

    if len(future_df) == 0:
        print("No future dates found in data. Check that weather forecasts are available.")
        return pd.DataFrame()

    # Limit forecast horizon
    max_forecast_date = historical_df['timestamp'].max() + pd.Timedelta(days=horizon_days)
    future_df = future_df[future_df['timestamp'] <= max_forecast_date].copy()

    print(f"Generating forecasts for {len(future_df)} hours...")

    # Load models
    print("Loading models...")
    redemption_model, sales_model = load_models()

    # Prepare data for redemption prediction
    forecast_redemption = prepare_forecast_data(
        future_df, historical_df, TARGET_REDEMPTION
    )

    # Prepare data for sales prediction
    forecast_sales = prepare_forecast_data(
        future_df, historical_df, TARGET_SALES
    )

    # Generate predictions
    print("Generating redemption predictions...")
    redemption_preds = redemption_model.predict(forecast_redemption)

    print("Generating sales predictions...")
    sales_preds = sales_model.predict(forecast_sales)

    # Build output DataFrame
    forecasts = pd.DataFrame({
        'timestamp': forecast_redemption['timestamp'].values,
        'predicted_redemption_count': redemption_preds,
        'predicted_sales_count': sales_preds,
    })

    # Add weather data for context
    weather_cols = ['temp', 'prcp', 'wspd', 'rhum']
    for col in weather_cols:
        if col in forecast_redemption.columns:
            forecasts[col] = forecast_redemption[col].values

    # Add temporal features for dashboard filtering
    forecasts['hour_of_day'] = pd.to_datetime(forecasts['timestamp']).dt.hour
    forecasts['day_of_week'] = pd.to_datetime(forecasts['timestamp']).dt.dayofweek
    forecasts['date'] = pd.to_datetime(forecasts['timestamp']).dt.date

    # Round predictions to integers (these are counts)
    forecasts['predicted_redemption_count'] = forecasts['predicted_redemption_count'].round().astype(int)
    forecasts['predicted_sales_count'] = forecasts['predicted_sales_count'].round().astype(int)

    # Ensure non-negative
    forecasts['predicted_redemption_count'] = forecasts['predicted_redemption_count'].clip(lower=0)
    forecasts['predicted_sales_count'] = forecasts['predicted_sales_count'].clip(lower=0)

    # Sort by timestamp
    forecasts = forecasts.sort_values('timestamp').reset_index(drop=True)

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_path, index=False)
    print(f"Forecasts saved to {output_path}")

    # Print summary
    print(f"\nForecast Summary:")
    print(f"  Period: {forecasts['timestamp'].min()} to {forecasts['timestamp'].max()}")
    print(f"  Total hours: {len(forecasts)}")
    print(f"  Total predicted redemptions: {forecasts['predicted_redemption_count'].sum():,}")
    print(f"  Total predicted sales: {forecasts['predicted_sales_count'].sum():,}")

    return forecasts


def create_daily_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily aggregated summary from hourly forecasts.

    Args:
        forecasts: Hourly forecast DataFrame

    Returns:
        Daily summary DataFrame
    """
    daily = forecasts.groupby('date').agg({
        'predicted_redemption_count': 'sum',
        'predicted_sales_count': 'sum',
        'temp': 'mean',
        'prcp': 'sum',
        'wspd': 'mean',
    }).reset_index()

    daily.columns = [
        'date', 'total_redemptions', 'total_sales',
        'avg_temp', 'total_precip', 'avg_wind'
    ]

    return daily


def generate_long_term_forecasts(
    features_path: str = None,
    output_path: str = None,
    horizon_days: int = LONG_TERM_HORIZON_DAYS
) -> pd.DataFrame:
    """
    Generate long-term (365-day) forecasts using blended approach.

    Days 1-7: Weather model with real forecasts
    Days 8-365: No-weather model with climatology
    Days 5-8: Blended transition

    Args:
        features_path: Path to features.csv
        output_path: Path to save forecasts
        horizon_days: Days to forecast (default: 365)

    Returns:
        DataFrame with long-term predictions
    """
    if features_path is None:
        features_path = FEATURES_CSV
    if output_path is None:
        output_path = LONG_TERM_FORECASTS_CSV

    print(f"Generating {horizon_days}-day long-term forecasts...")

    # Load full dataset
    df = pd.read_csv(features_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Aggregate to hourly
    hourly_df = aggregate_to_hourly(df)

    # Split into historical and future
    historical_mask = hourly_df[TARGET_REDEMPTION].notna()
    historical_df = hourly_df[historical_mask].copy()

    # Get last date with actual data
    last_actual_date = historical_df['timestamp'].max()
    print(f"Last actual data: {last_actual_date}")

    # Load climatology for weather features beyond 7 days
    print("Loading weather climatology...")
    climatology_df = load_or_compute_climatology()

    # Generate future date range (365 days)
    start_date = last_actual_date + pd.Timedelta(hours=1)
    end_date = last_actual_date + pd.Timedelta(days=horizon_days)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='h')

    print(f"Forecast period: {start_date} to {end_date}")
    print(f"Total hours to forecast: {len(future_dates)}")

    # Load both model types
    print("Loading models...")
    try:
        redemption_weather, sales_weather = load_models()
        redemption_no_weather, sales_no_weather = load_no_weather_models()
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Make sure to train both weather and no-weather models first.")
        return pd.DataFrame()

    # Build future DataFrame
    future_df = pd.DataFrame({'timestamp': future_dates})
    future_df['hour_of_day'] = future_df['timestamp'].dt.hour
    future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['month'] = future_df['timestamp'].dt.month
    future_df['day_of_month'] = future_df['timestamp'].dt.day
    future_df['week_of_year'] = future_df['timestamp'].dt.isocalendar().week.astype(int)
    future_df['year'] = future_df['timestamp'].dt.year
    future_df['date'] = future_df['timestamp'].dt.date

    # Compute forecast day (days from last actual)
    future_df['forecast_day'] = ((future_df['timestamp'] - last_actual_date).dt.total_seconds() / 86400).astype(int) + 1

    # Add weather features: real forecasts for days 1-7, climatology for days 8+
    # First, check if we have any real weather forecasts in the original data
    future_in_data = hourly_df[~historical_mask].copy()

    for idx, row in future_df.iterrows():
        forecast_day = row['forecast_day']
        ts = row['timestamp']

        if forecast_day <= 7 and len(future_in_data) > 0:
            # Use real weather forecast if available
            matching = future_in_data[future_in_data['timestamp'] == ts]
            if len(matching) > 0:
                for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wspd', 'wdir', 'pres', 'coco']:
                    if col in matching.columns:
                        future_df.loc[idx, col] = matching[col].values[0]
                continue

        # Use climatology for this datetime
        doy = ts.timetuple().tm_yday
        hour = ts.hour
        clim_row = climatology_df[
            (climatology_df['day_of_year'] == doy) &
            (climatology_df['hour_of_day'] == hour)
        ]
        if len(clim_row) > 0:
            for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wspd', 'wdir', 'pres', 'coco']:
                mean_col = f'{col}_mean'
                if mean_col in clim_row.columns:
                    future_df.loc[idx, col] = clim_row[mean_col].values[0]

    # Fill any remaining NaN weather values
    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd', 'wdir', 'pres', 'coco']
    for col in weather_cols:
        if col in future_df.columns:
            future_df[col] = future_df[col].ffill().bfill()

    # Add calendar features (holidays, school breaks, etc.)
    # These are deterministic and can be computed for any future date
    future_df['is_holiday'] = 0
    future_df['holiday_label'] = 11  # "No Holiday" encoding
    future_df['is_school_break'] = 0
    future_df['is_covid_lockdown'] = 0
    future_df['is_flooding'] = 0

    # Compute days since/until weekend
    future_df['days_since_weekend'] = future_df['day_of_week'].apply(
        lambda x: 0 if x >= 5 else x + 2 if x == 0 else x + 1
    )
    future_df['days_until_weekend'] = future_df['day_of_week'].apply(
        lambda x: 0 if x >= 5 else 5 - x
    )

    # Add target columns as NaN (for lag computation)
    future_df[TARGET_REDEMPTION] = np.nan
    future_df[TARGET_SALES] = np.nan

    # For long-term forecasts, compute lag features using SAME PERIOD FROM PREVIOUS YEAR
    # This is much more meaningful than using last week's data for a July forecast
    print("Computing lag features from previous year data...")

    # Create lookup for historical data by day-of-year and hour
    historical_df['day_of_year'] = historical_df['timestamp'].dt.dayofyear
    historical_df['hour_of_day'] = historical_df['timestamp'].dt.hour

    # Get previous year's data for lag computation
    prev_year = last_actual_date.year - 1
    prev_year_data = historical_df[historical_df['timestamp'].dt.year == prev_year].copy()

    # Create lookup dictionary for fast access
    redemption_lookup = {}
    sales_lookup = {}
    for _, row in prev_year_data.iterrows():
        key = (row['day_of_year'], row['hour_of_day'])
        redemption_lookup[key] = row[TARGET_REDEMPTION]
        sales_lookup[key] = row[TARGET_SALES]

    # Compute lag features for future_df using previous year's same day/hour
    future_df['day_of_year'] = future_df['timestamp'].dt.dayofyear

    lag_redemption = []
    lag_sales = []
    rolling_mean_redemption = []
    rolling_mean_sales = []

    for _, row in future_df.iterrows():
        doy = row['day_of_year']
        hour = row['hour_of_day']
        key = (doy, hour)

        # Get value from same day/hour last year
        red_val = redemption_lookup.get(key, 0)
        sales_val = sales_lookup.get(key, 0)

        lag_redemption.append(red_val)
        lag_sales.append(sales_val)

        # For rolling mean, use average of surrounding hours from previous year
        rolling_vals_red = []
        rolling_vals_sales = []
        for h_offset in range(-12, 13):  # +/- 12 hours
            h = (hour + h_offset) % 24
            k = (doy, h)
            if k in redemption_lookup:
                rolling_vals_red.append(redemption_lookup[k])
                rolling_vals_sales.append(sales_lookup[k])

        rolling_mean_redemption.append(np.mean(rolling_vals_red) if rolling_vals_red else red_val)
        rolling_mean_sales.append(np.mean(rolling_vals_sales) if rolling_vals_sales else sales_val)

    future_df[f'{TARGET_REDEMPTION}_lag_168h'] = lag_redemption
    future_df[f'{TARGET_SALES}_lag_168h'] = lag_sales
    future_df[f'{TARGET_REDEMPTION}_rolling_mean_168h'] = rolling_mean_redemption
    future_df[f'{TARGET_SALES}_rolling_mean_168h'] = rolling_mean_sales

    # Compute rolling std from the rolling values
    future_df[f'{TARGET_REDEMPTION}_rolling_std_168h'] = future_df[f'{TARGET_REDEMPTION}_rolling_mean_168h'] * 0.3
    future_df[f'{TARGET_SALES}_rolling_std_168h'] = future_df[f'{TARGET_SALES}_rolling_mean_168h'] * 0.3

    # Also add short-term lags (use same values as 168h lag for consistency)
    future_df[f'{TARGET_REDEMPTION}_lag_1h'] = lag_redemption
    future_df[f'{TARGET_REDEMPTION}_lag_6h'] = lag_redemption
    future_df[f'{TARGET_REDEMPTION}_lag_24h'] = lag_redemption
    future_df[f'{TARGET_SALES}_lag_1h'] = lag_sales
    future_df[f'{TARGET_SALES}_lag_6h'] = lag_sales
    future_df[f'{TARGET_SALES}_lag_24h'] = lag_sales

    # Add rolling 24h features (use same as 168h for long-term)
    future_df[f'{TARGET_REDEMPTION}_rolling_mean_24h'] = rolling_mean_redemption
    future_df[f'{TARGET_REDEMPTION}_rolling_std_24h'] = future_df[f'{TARGET_REDEMPTION}_rolling_std_168h']
    future_df[f'{TARGET_SALES}_rolling_mean_24h'] = rolling_mean_sales
    future_df[f'{TARGET_SALES}_rolling_std_24h'] = future_df[f'{TARGET_SALES}_rolling_std_168h']

    # Add cyclical features
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour_of_day'] / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour_of_day'] / 24)
    future_df['dow_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['dow_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

    # Add season encoding
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    future_df['season_encoded'] = future_df['month'].apply(get_season)

    # Add tourism and World Cup features
    print("Adding tourism and World Cup features...")
    future_df = add_tourism_features(future_df)

    # Use future_df directly as forecast data (already has all features)
    forecast_redemption_weather = future_df.copy()
    forecast_sales_weather = future_df.copy()

    # For no-weather models, use the same prepared data
    forecast_redemption_no_weather = forecast_redemption_weather.copy()
    forecast_sales_no_weather = forecast_sales_weather.copy()

    # Generate predictions from both models
    print("Generating weather model predictions...")
    redemption_preds_weather = redemption_weather.predict(forecast_redemption_weather)
    sales_preds_weather = sales_weather.predict(forecast_sales_weather)

    print("Generating no-weather model predictions...")
    redemption_preds_no_weather = redemption_no_weather.predict(forecast_redemption_no_weather)
    sales_preds_no_weather = sales_no_weather.predict(forecast_sales_no_weather)

    # Blend predictions based on forecast day
    print("Blending predictions...")
    forecast_days = forecast_redemption_weather['forecast_day'].values if 'forecast_day' in forecast_redemption_weather.columns else future_df['forecast_day'].values

    blended_redemption = np.array([
        blend_predictions(w, nw, fd)
        for w, nw, fd in zip(redemption_preds_weather, redemption_preds_no_weather, forecast_days)
    ])
    blended_sales = np.array([
        blend_predictions(w, nw, fd)
        for w, nw, fd in zip(sales_preds_weather, sales_preds_no_weather, forecast_days)
    ])

    # Compute prediction intervals
    redemption_intervals = [
        compute_prediction_interval(p, fd) for p, fd in zip(blended_redemption, forecast_days)
    ]
    sales_intervals = [
        compute_prediction_interval(p, fd) for p, fd in zip(blended_sales, forecast_days)
    ]

    # Round predictions to integers (no scaling factor - tourism features handle this)
    final_redemption = blended_redemption.round().astype(int)
    final_sales = blended_sales.round().astype(int)

    # Build output DataFrame
    forecasts = pd.DataFrame({
        'timestamp': forecast_redemption_weather['timestamp'].values,
        'predicted_redemption_count': final_redemption,
        'predicted_sales_count': final_sales,
        'redemption_lower': [int(max(0, x[0])) for x in redemption_intervals],
        'redemption_upper': [int(x[1]) for x in redemption_intervals],
        'sales_lower': [int(max(0, x[0])) for x in sales_intervals],
        'sales_upper': [int(x[1]) for x in sales_intervals],
        'forecast_day': forecast_days,
    })

    # Determine forecast type
    forecasts['forecast_type'] = forecasts['forecast_day'].apply(
        lambda d: 'weather' if d <= BLEND_TRANSITION_START
        else 'blended' if d < BLEND_TRANSITION_END
        else 'climatology'
    )

    # Add weather and temporal features
    for col in ['temp', 'prcp', 'wspd']:
        if col in forecast_redemption_weather.columns:
            forecasts[col] = forecast_redemption_weather[col].values

    forecasts['hour_of_day'] = pd.to_datetime(forecasts['timestamp']).dt.hour
    forecasts['day_of_week'] = pd.to_datetime(forecasts['timestamp']).dt.dayofweek
    forecasts['date'] = pd.to_datetime(forecasts['timestamp']).dt.date
    forecasts['month'] = pd.to_datetime(forecasts['timestamp']).dt.month
    forecasts['year'] = pd.to_datetime(forecasts['timestamp']).dt.year

    # Ensure non-negative
    forecasts['predicted_redemption_count'] = forecasts['predicted_redemption_count'].clip(lower=0)
    forecasts['predicted_sales_count'] = forecasts['predicted_sales_count'].clip(lower=0)

    # Sort by timestamp
    forecasts = forecasts.sort_values('timestamp').reset_index(drop=True)

    # Save hourly forecasts
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_path, index=False)
    print(f"Long-term forecasts saved to {output_path}")

    # Create and save daily summary
    daily_summary = create_long_term_daily_summary(forecasts)
    daily_summary.to_csv(LONG_TERM_DAILY_CSV, index=False)
    print(f"Daily summary saved to {LONG_TERM_DAILY_CSV}")

    # Create and save monthly summary
    monthly_summary = create_monthly_summary(forecasts)
    monthly_summary.to_csv(LONG_TERM_MONTHLY_CSV, index=False)
    print(f"Monthly summary saved to {LONG_TERM_MONTHLY_CSV}")

    # Print summary
    print(f"\nLong-Term Forecast Summary:")
    print(f"  Period: {forecasts['timestamp'].min()} to {forecasts['timestamp'].max()}")
    print(f"  Total hours: {len(forecasts)}")
    print(f"  Total predicted redemptions: {forecasts['predicted_redemption_count'].sum():,}")
    print(f"  Total predicted sales: {forecasts['predicted_sales_count'].sum():,}")

    return forecasts


def create_long_term_daily_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily summary from long-term forecasts including confidence intervals.

    Args:
        forecasts: Long-term hourly forecast DataFrame

    Returns:
        Daily summary DataFrame
    """
    agg_dict = {
        'predicted_redemption_count': 'sum',
        'predicted_sales_count': 'sum',
        'redemption_lower': 'sum',
        'redemption_upper': 'sum',
        'sales_lower': 'sum',
        'sales_upper': 'sum',
        'forecast_day': 'first',
        'forecast_type': 'first',
    }

    # Add weather columns if present
    for col in ['temp', 'prcp', 'wspd']:
        if col in forecasts.columns:
            agg_dict[col] = 'mean' if col != 'prcp' else 'sum'

    daily = forecasts.groupby('date').agg(agg_dict).reset_index()

    # Rename columns
    daily = daily.rename(columns={
        'predicted_redemption_count': 'total_redemptions',
        'predicted_sales_count': 'total_sales',
        'temp': 'avg_temp',
        'prcp': 'total_precip',
        'wspd': 'avg_wind',
    })

    return daily


def create_monthly_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly summary from long-term forecasts.

    Args:
        forecasts: Long-term hourly forecast DataFrame

    Returns:
        Monthly summary DataFrame
    """
    forecasts = forecasts.copy()
    forecasts['year_month'] = pd.to_datetime(forecasts['timestamp']).dt.to_period('M').astype(str)

    agg_dict = {
        'predicted_redemption_count': 'sum',
        'predicted_sales_count': 'sum',
        'redemption_lower': 'sum',
        'redemption_upper': 'sum',
        'sales_lower': 'sum',
        'sales_upper': 'sum',
    }

    # Add weather columns if present
    for col in ['temp', 'prcp']:
        if col in forecasts.columns:
            agg_dict[col] = 'mean' if col == 'temp' else 'sum'

    monthly = forecasts.groupby('year_month').agg(agg_dict).reset_index()

    # Rename columns
    monthly = monthly.rename(columns={
        'predicted_redemption_count': 'total_redemptions',
        'predicted_sales_count': 'total_sales',
        'temp': 'avg_temp',
        'prcp': 'total_precip',
    })

    # Add forecast confidence indicator based on how far into the future
    # First month = higher confidence, later months = lower
    monthly['forecast_confidence'] = 'high'
    monthly.loc[monthly.index > 0, 'forecast_confidence'] = 'medium'
    monthly.loc[monthly.index > 2, 'forecast_confidence'] = 'low'

    return monthly
