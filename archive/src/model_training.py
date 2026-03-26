"""
Model training and loading for Toronto Ferry ridership forecasting.
Uses LightGBM for gradient boosting regression.
"""
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import (
    CATEGORICAL_FEATURES, MODEL_PARAMS, WEATHER_FEATURES,
    TEMPORAL_FEATURES, CALENDAR_EVENT_FEATURES,
    TARGET_REDEMPTION, TARGET_SALES, TEST_MONTHS, VAL_MONTHS,
    REDEMPTION_MODEL_PATH, SALES_MODEL_PATH,
    REDEMPTION_NO_WEATHER_MODEL_PATH, SALES_NO_WEATHER_MODEL_PATH,
    MODEL_METADATA_PATH, NO_WEATHER_FEATURES, NO_WEATHER_LAG_FEATURES_TEMPLATE,
    TOURISM_FEATURES,
)
from .feature_engineering import (
    prepare_features, get_feature_columns, split_train_test_by_time,
    aggregate_to_hourly,
)


class FerryRidershipModel:
    """Wrapper around LGBMRegressor for ferry ridership prediction."""

    def __init__(
        self,
        model: LGBMRegressor,
        feature_columns: List[str],
        categorical_features: List[str],
        target: str,
        params: Dict,
        metrics: Optional[Dict] = None,
        training_date: Optional[str] = None,
        include_weather: bool = True,
        feature_importance: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        self.target = target
        self.params = params
        self.metrics = metrics or {}
        self.training_date = training_date or datetime.now().isoformat()
        self.include_weather = include_weather
        self.feature_importance = feature_importance

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions, selecting only the expected feature columns."""
        available = [c for c in self.feature_columns if c in df.columns]
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            # Fill missing columns with 0
            for col in missing:
                df = df.copy()
                df[col] = 0
            available = self.feature_columns

        X = df[self.feature_columns].copy()
        # Ensure categorical columns are int
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].fillna(0).astype(int)
        X = X.fillna(0)
        # Use numpy array to avoid LightGBM categorical dtype mismatch
        preds = self.model.predict(X.values)
        return np.clip(preds, 0, None)

    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'target': self.target,
            'params': self.params,
            'metrics': self.metrics,
            'training_date': self.training_date,
            'include_weather': self.include_weather,
            'feature_importance': (
                self.feature_importance.to_dict()
                if isinstance(self.feature_importance, pd.DataFrame)
                else self.feature_importance
            ),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'FerryRidershipModel':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        fi = data.get('feature_importance')
        if isinstance(fi, dict) and 'feature' in fi and 'importance' in fi:
            fi = pd.DataFrame(fi).sort_values('importance', ascending=False).reset_index(drop=True)

        return cls(
            model=data['model'],
            feature_columns=data['feature_columns'],
            categorical_features=data.get('categorical_features', []),
            target=data['target'],
            params=data.get('params', {}),
            metrics=data.get('metrics', {}),
            training_date=data.get('training_date', ''),
            include_weather=data.get('include_weather', True),
            feature_importance=fi,
        )


def _train_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    cat_features: List[str],
    include_weather: bool = True,
) -> Tuple['FerryRidershipModel', Dict]:
    """Train a single LightGBM model."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

    # Prepare features
    df_prepared = prepare_features(df, target_col=target_col)
    hourly = aggregate_to_hourly(df_prepared) if 'timestamp_hour' not in df_prepared.columns else df_prepared

    # Drop rows with NaN target
    hourly = hourly.dropna(subset=[target_col])

    # Split
    train_df, val_df, test_df = split_train_test_by_time(hourly, TEST_MONTHS, VAL_MONTHS)

    # Get feature columns
    available_features = [c for c in feature_cols if c in hourly.columns]
    available_cats = [c for c in cat_features if c in available_features]

    X_train = train_df[available_features].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[available_features].fillna(0)
    y_val = val_df[target_col]
    X_test = test_df[available_features].fillna(0)
    y_test = test_df[target_col]

    for col in available_cats:
        X_train[col] = X_train[col].astype(int)
        X_val[col] = X_val[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    # Train
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=available_cats,
    )

    # Evaluate
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    metrics = {
        'mae': mean_absolute_error(y_test, preds),
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'r2': r2_score(y_test, preds),
        'median_ae': median_absolute_error(y_test, preds),
        'n_test_samples': len(y_test),
    }

    # Feature importance
    fi_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    wrapped = FerryRidershipModel(
        model=model,
        feature_columns=available_features,
        categorical_features=available_cats,
        target=target_col,
        params=MODEL_PARAMS,
        metrics=metrics,
        include_weather=include_weather,
        feature_importance=fi_df,
    )

    return wrapped, metrics


def train_both_models(df: pd.DataFrame) -> Tuple['FerryRidershipModel', 'FerryRidershipModel', Dict]:
    """Train both redemption and sales weather models."""
    feature_cols = get_feature_columns(df, TARGET_REDEMPTION)

    print("Training redemption model...")
    redemption_model, red_metrics = _train_model(
        df, TARGET_REDEMPTION, feature_cols, CATEGORICAL_FEATURES, include_weather=True
    )
    redemption_model.save(REDEMPTION_MODEL_PATH)
    print(f"  Saved to {REDEMPTION_MODEL_PATH}")

    print("Training sales model...")
    sales_model, sales_metrics = _train_model(
        df, TARGET_SALES, feature_cols, CATEGORICAL_FEATURES, include_weather=True
    )
    sales_model.save(SALES_MODEL_PATH)
    print(f"  Saved to {SALES_MODEL_PATH}")

    return redemption_model, sales_model, {
        'redemption': red_metrics,
        'sales': sales_metrics,
    }


def train_no_weather_models(df: pd.DataFrame) -> Tuple['FerryRidershipModel', 'FerryRidershipModel', Dict]:
    """Train both no-weather models for long-term forecasting."""
    # Build feature list (no-weather features + lag features for each target)
    red_lag_features = [t.format(target=TARGET_REDEMPTION) for t in NO_WEATHER_LAG_FEATURES_TEMPLATE]
    sales_lag_features = [t.format(target=TARGET_SALES) for t in NO_WEATHER_LAG_FEATURES_TEMPLATE]
    no_weather_cats = [c for c in CATEGORICAL_FEATURES if c not in ['coco']]

    print("Training no-weather redemption model...")
    red_features = NO_WEATHER_FEATURES + red_lag_features
    redemption_model, red_metrics = _train_model(
        df, TARGET_REDEMPTION, red_features, no_weather_cats, include_weather=False
    )
    redemption_model.save(REDEMPTION_NO_WEATHER_MODEL_PATH)
    print(f"  Saved to {REDEMPTION_NO_WEATHER_MODEL_PATH}")

    print("Training no-weather sales model...")
    sales_features = NO_WEATHER_FEATURES + sales_lag_features
    sales_model, sales_metrics = _train_model(
        df, TARGET_SALES, sales_features, no_weather_cats, include_weather=False
    )
    sales_model.save(SALES_NO_WEATHER_MODEL_PATH)
    print(f"  Saved to {SALES_NO_WEATHER_MODEL_PATH}")

    return redemption_model, sales_model, {
        'redemption_no_weather': red_metrics,
        'sales_no_weather': sales_metrics,
    }
