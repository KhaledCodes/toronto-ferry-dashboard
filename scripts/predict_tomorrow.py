"""
Train a daily LightGBM model and predict tomorrow's ferry ridership.
Outputs: outputs/prediction.json
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import holidays


ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = ROOT / "outputs" / "daily_totals.csv"
OUTPUT_JSON = ROOT / "outputs" / "prediction.json"
HISTORY_CSV = ROOT / "outputs" / "prediction_history.csv"

FEATURES = [
    "day_of_week", "month", "day_of_month", "week_of_year", "year", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "lag_1d", "lag_7d", "lag_14d", "lag_28d", "lag_365d",
    "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean", "rolling_30d_mean",
    "is_holiday",
]

ON_HOLIDAYS = holidays.Canada(prov="ON")


def add_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    doy = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366)

    df["lag_1d"] = df["redemptions"].shift(1)
    df["lag_7d"] = df["redemptions"].shift(7)
    df["lag_14d"] = df["redemptions"].shift(14)
    df["lag_28d"] = df["redemptions"].shift(28)
    df["lag_365d"] = df["redemptions"].shift(365)

    df["rolling_7d_mean"] = df["redemptions"].shift(1).rolling(7).mean()
    df["rolling_7d_std"] = df["redemptions"].shift(1).rolling(7).std()
    df["rolling_14d_mean"] = df["redemptions"].shift(1).rolling(14).mean()
    df["rolling_30d_mean"] = df["redemptions"].shift(1).rolling(30).mean()

    df["is_holiday"] = df["date"].apply(lambda d: int(d in ON_HOLIDAYS))

    return df


def train_and_predict(df):
    featured = add_features(df)
    featured = featured.dropna(subset=FEATURES)

    train = featured.iloc[:-30]
    val = featured.iloc[-30:]

    X_train, y_train = train[FEATURES], train["redemptions"]
    X_val, y_val = val[FEATURES], val["redemptions"]

    model = LGBMRegressor(
        objective="poisson",
        metric="rmse",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        verbose=-1,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )

    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    residual_std = np.std(y_val.values - val_pred)

    tomorrow = df["date"].max() + timedelta(days=1)
    tomorrow_row = pd.DataFrame([{"date": tomorrow, "redemptions": np.nan}])
    extended = pd.concat([df, tomorrow_row], ignore_index=True)
    extended = add_features(extended)
    tomorrow_features = extended[extended["date"] == tomorrow][FEATURES]

    prediction = max(0, round(model.predict(tomorrow_features)[0]))
    ci_lower = max(0, round(prediction - 1.28 * residual_std))
    ci_upper = round(prediction + 1.28 * residual_std)

    return model, prediction, ci_lower, ci_upper, mae, r2, tomorrow, residual_std


def build_context(df, tomorrow):
    last_7 = df.tail(7)
    last_7_list = [
        {"date": row["date"].strftime("%Y-%m-%d"), "redemptions": int(row["redemptions"])}
        for _, row in last_7.iterrows()
    ]

    dow = tomorrow.weekday()
    same_dow = df[df["date"].dt.dayofweek == dow].tail(8)
    same_dow_avg = round(same_dow["redemptions"].mean())

    return {
        "same_dow_avg": same_dow_avg,
        "last_7d": last_7_list,
    }


def update_history(df, prediction, ci_lower, ci_upper, tomorrow):
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    actuals = df.set_index("date")["redemptions"]

    if HISTORY_CSV.exists():
        history = pd.read_csv(HISTORY_CSV)
    else:
        history = pd.DataFrame(columns=["date", "day", "predicted", "ci_lower", "ci_upper", "actual", "error_pct"])

    # Only treat a date as final once a LATER day exists in the data. The most
    # recent day is typically still being collected and reports a partial
    # (often 0) total; finalizing it early locks in a wrong actual that never
    # gets corrected.
    latest_date = df["date"].max()

    if not history.empty:
        for i, row in history.iterrows():
            existing = row.get("actual")
            # Re-evaluate rows that are unset OR were previously locked at a
            # non-positive (incomplete) value.
            if not (pd.isna(existing) or existing == "" or float(existing) <= 0):
                continue

            d = pd.Timestamp(row["date"])
            if d >= latest_date or d not in actuals.index:
                continue  # day not settled yet — retry on a later run

            actual = int(actuals[d])
            if actual <= 0:
                continue  # still looks incomplete

            history.at[i, "actual"] = actual
            predicted = int(row["predicted"])
            history.at[i, "error_pct"] = round(abs(predicted - actual) / actual * 100, 1)

    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    if history.empty or tomorrow_str not in history["date"].values:
        new_row = {
            "date": tomorrow_str,
            "day": day_names[tomorrow.weekday()],
            "predicted": prediction,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "actual": "",
            "error_pct": "",
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    history.to_csv(HISTORY_CSV, index=False)
    return history


def main():
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    model, prediction, ci_lower, ci_upper, mae, r2, tomorrow, residual_std = train_and_predict(df)
    context = build_context(df, tomorrow)
    update_history(df, prediction, ci_lower, ci_upper, tomorrow)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    output = {
        "prediction_date": tomorrow.strftime("%Y-%m-%d"),
        "day_of_week_name": day_names[tomorrow.weekday()],
        "predicted_redemptions": prediction,
        "confidence_interval": {
            "lower": ci_lower,
            "upper": ci_upper,
            "level": 0.80,
        },
        "context": context,
        "model_info": {
            "validation_mae": round(mae),
            "validation_r2": round(r2, 3),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Prediction for {tomorrow.strftime('%A, %B %d, %Y')}: {prediction:,}")
    print(f"  80% CI: [{ci_lower:,} - {ci_upper:,}]")
    print(f"  Validation MAE: {mae:.0f}, R²: {r2:.3f}")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"  History: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
