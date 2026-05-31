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

# Upper bound on how many days ahead we recursively forecast to backfill skipped
# days. Bounds runaway if the upstream feed is frozen for a long stretch.
MAX_FORECAST_DAYS = 14

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


def train(df):
    featured = add_features(df)
    featured = featured.dropna(subset=FEATURES)

    train_set = featured.iloc[:-30]
    val = featured.iloc[-30:]

    X_train, y_train = train_set[FEATURES], train_set["redemptions"]
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

    return model, mae, r2, residual_std


def forecast_horizon(model, df, residual_std, end_date):
    """Recursively forecast each day from (last data day + 1) through end_date.

    Each day's prediction is fed back in as the lag basis for the next day, so a
    multi-day span is forecast without ever using a future actual — there is no
    leakage. This guarantees every elapsed calendar day gets a logged prediction
    even when the upstream feed jumps forward several days in one step. The
    horizon always covers at least the genuine one-step-ahead day.
    """
    data_max = df["date"].max()
    horizon_end = max(pd.Timestamp(end_date), data_max + timedelta(days=1))
    horizon_end = min(horizon_end, data_max + timedelta(days=MAX_FORECAST_DAYS))

    work = df[["date", "redemptions"]].copy()
    forecasts = []
    cur = data_max + timedelta(days=1)
    while cur <= horizon_end:
        extended = pd.concat(
            [work, pd.DataFrame([{"date": cur, "redemptions": np.nan}])],
            ignore_index=True,
        )
        feats = add_features(extended)
        cur_features = feats[feats["date"] == cur][FEATURES]

        prediction = max(0, round(model.predict(cur_features)[0]))
        ci_lower = max(0, round(prediction - 1.28 * residual_std))
        ci_upper = round(prediction + 1.28 * residual_std)
        forecasts.append({
            "date": cur,
            "prediction": prediction,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

        # Feed this prediction back in so the next day's lag features see it.
        work = pd.concat(
            [work, pd.DataFrame([{"date": cur, "redemptions": prediction}])],
            ignore_index=True,
        )
        cur += timedelta(days=1)

    return forecasts


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


def update_history(df, forecasts):
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    actuals = df.set_index("date")["redemptions"]

    if HISTORY_CSV.exists():
        history = pd.read_csv(HISTORY_CSV)
    else:
        history = pd.DataFrame(columns=["date", "day", "predicted", "ci_lower", "ci_upper", "actual", "error_pct"])

    # 1. Finalize predictions for days that are now complete. A day is complete
    #    once a LATER day exists in the data: the feed fills chronologically, so
    #    a newer timestamp can only appear after every earlier interval — meaning
    #    the previous day is fully present. The most recent day in the data may
    #    still be collecting, so we never finalize it.
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

    # 2. Log a prediction row for every forecast day not already tracked. The
    #    horizon spans (last data day + 1) .. today, so a multi-day feed jump
    #    never leaves an elapsed day without a logged prediction. Existing rows
    #    are left untouched — a day's prediction is logged once, when first made.
    existing_dates = set(history["date"].astype(str)) if not history.empty else set()
    new_rows = []
    for fc in forecasts:
        date_str = fc["date"].strftime("%Y-%m-%d")
        if date_str in existing_dates:
            continue
        new_rows.append({
            "date": date_str,
            "day": day_names[fc["date"].weekday()],
            "predicted": fc["prediction"],
            "ci_lower": fc["ci_lower"],
            "ci_upper": fc["ci_upper"],
            "actual": "",
            "error_pct": "",
        })

    if new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)

    history = history.sort_values("date").reset_index(drop=True)
    history.to_csv(HISTORY_CSV, index=False)
    return history


def main():
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    model, mae, r2, residual_std = train(df)

    # Forecast from the last data day forward through today, so any days the
    # feed skipped over still get a logged prediction.
    today = pd.Timestamp(datetime.utcnow().date())
    forecasts = forecast_horizon(model, df, residual_std, today)

    # The headline forecast is the genuine one-step-ahead day (last data + 1);
    # any later days in the horizon exist only to backfill skipped days.
    headline = forecasts[0]
    tomorrow = headline["date"]

    context = build_context(df, tomorrow)
    update_history(df, forecasts)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    output = {
        "prediction_date": tomorrow.strftime("%Y-%m-%d"),
        "day_of_week_name": day_names[tomorrow.weekday()],
        "predicted_redemptions": headline["prediction"],
        "confidence_interval": {
            "lower": headline["ci_lower"],
            "upper": headline["ci_upper"],
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

    print(f"Prediction for {tomorrow.strftime('%A, %B %d, %Y')}: {headline['prediction']:,}")
    print(f"  80% CI: [{headline['ci_lower']:,} - {headline['ci_upper']:,}]")
    print(f"  Validation MAE: {mae:.0f}, R²: {r2:.3f}")
    if len(forecasts) > 1:
        print(f"  Backfilled {len(forecasts) - 1} skipped day(s) through {forecasts[-1]['date'].date()}")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"  History: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
