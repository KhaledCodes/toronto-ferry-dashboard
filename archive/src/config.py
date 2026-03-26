"""
Central configuration for the Toronto Ferry ridership forecasting system.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
FEATURES_CSV = DATA_DIR / "features.csv"
FERRY_DATA_CSV = DATA_DIR / "ferry_data.csv"
FORECASTS_CSV = OUTPUTS_DIR / "forecasts.csv"
HOURLY_DATA_CSV = OUTPUTS_DIR / "hourly_data.csv"

# Model files
REDEMPTION_MODEL_PATH = MODELS_DIR / "redemption_model.pkl"
SALES_MODEL_PATH = MODELS_DIR / "sales_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Model hyperparameters (proven to work from existing notebook)
MODEL_PARAMS = {
    'objective': 'poisson',  # Good for count data (non-negative integers)
    'metric': 'rmse',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

# Lag periods (in hours, since we aggregate to hourly)
LAG_PERIODS = [1, 6, 24, 168]  # 1h, 6h, 1day, 1week

# Rolling window sizes (in hours)
ROLLING_WINDOWS = [24, 168]  # 1day, 1week

# Feature lists
WEATHER_FEATURES = [
    'temp',   # Temperature
    'dwpt',   # Dew point
    'rhum',   # Relative humidity
    'prcp',   # Precipitation
    'wspd',   # Wind speed
    'wdir',   # Wind direction
    'pres',   # Atmospheric pressure
    'coco',   # Cloud cover code
]

TEMPORAL_FEATURES = [
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'month',
    'day_of_month',
    'week_of_year',
    'year',
]

CALENDAR_EVENT_FEATURES = [
    'is_holiday',
    'holiday_label',
    'is_school_break',
    'is_covid_lockdown',
    'is_flooding',
    'days_since_weekend',
    'days_until_weekend',
]

# Categorical features for LightGBM
CATEGORICAL_FEATURES = [
    'coco',
    'day_of_week',
    'is_weekend',
    'month',
    'hour_of_day',
    'is_holiday',
    'is_school_break',
    'is_covid_lockdown',
    'is_flooding',
    'holiday_label',
    'is_world_cup_period',
    'is_world_cup_match_day',
    'is_canada_opening',
]

# Target columns
TARGET_REDEMPTION = 'redemption_count'
TARGET_SALES = 'sales_count'

# Forecast configuration
FORECAST_HORIZON_DAYS = 7
LONG_TERM_HORIZON_DAYS = 365

# Blending configuration for long-term forecasts
BLEND_TRANSITION_START = 5  # Days where blending starts
BLEND_TRANSITION_END = 8    # Days where blending ends (100% no-weather model)

# Long-term forecast output files
LONG_TERM_FORECASTS_CSV = OUTPUTS_DIR / "long_term_forecasts.csv"
LONG_TERM_DAILY_CSV = OUTPUTS_DIR / "long_term_daily_forecasts.csv"
LONG_TERM_MONTHLY_CSV = OUTPUTS_DIR / "long_term_monthly_forecasts.csv"
WEATHER_CLIMATOLOGY_CSV = OUTPUTS_DIR / "weather_climatology.csv"

# No-weather model files
REDEMPTION_NO_WEATHER_MODEL_PATH = MODELS_DIR / "redemption_model_no_weather.pkl"
SALES_NO_WEATHER_MODEL_PATH = MODELS_DIR / "sales_model_no_weather.pkl"

# Features for no-weather model (temporal, calendar, and tourism)
NO_WEATHER_FEATURES = [
    # Temporal
    'hour_of_day', 'day_of_week', 'is_weekend', 'month',
    'day_of_month', 'week_of_year', 'year',
    # Calendar
    'is_holiday', 'holiday_label', 'is_school_break',
    'is_covid_lockdown', 'is_flooding',
    'days_since_weekend', 'days_until_weekend',
    # Cyclical encodings
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'season_encoded',
    # Tourism features
    'tourism_index',
    'is_world_cup_period', 'is_world_cup_match_day', 'is_canada_opening',
    'days_to_world_cup', 'days_into_world_cup', 'world_cup_boost',
]

# Lag features to use for no-weather model (only 1-week lag available for long-term)
NO_WEATHER_LAG_FEATURES_TEMPLATE = [
    '{target}_lag_168h',
    '{target}_rolling_mean_168h',
    '{target}_rolling_std_168h',
]

# Train/validation/test split
TEST_MONTHS = 1
VAL_MONTHS = 1

# Tourism features
TOURISM_FEATURES = [
    'tourism_index',
    'is_world_cup_period',
    'is_world_cup_match_day',
    'is_canada_opening',
    'days_to_world_cup',
    'days_into_world_cup',
    'world_cup_boost',
]

# Categorical tourism features for LightGBM
TOURISM_CATEGORICAL_FEATURES = [
    'is_world_cup_period',
    'is_world_cup_match_day',
    'is_canada_opening',
]
