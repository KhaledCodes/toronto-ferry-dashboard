"""
Train a daily LightGBM model and predict tomorrow's ferry ridership.
Outputs: outputs/prediction.json
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import holidays


ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = ROOT / "outputs" / "daily_totals.csv"
HOURLY_CSV = ROOT / "outputs" / "hourly_totals.csv"
WEATHER_CSV = ROOT / "outputs" / "weather_daily.csv"
OUTPUT_JSON = ROOT / "outputs" / "prediction.json"
HISTORY_CSV = ROOT / "outputs" / "prediction_history.csv"

# Upper bound on how many days ahead we recursively forecast to backfill skipped
# days. Bounds runaway if the upstream feed is frozen for a long stretch.
MAX_FORECAST_DAYS = 14

# "Today" is judged in Toronto local time — the data timestamps and the dashboard
# audience are both local, so "tomorrow" must be the local calendar day.
LOCAL_TZ = ZoneInfo("America/Toronto")

# A day's intraday data is considered complete once it reaches this hour.
DAY_COMPLETE_HOUR = 23

BASE_FEATURES = [
    "day_of_week", "month", "day_of_month", "week_of_year", "year", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "lag_1d", "lag_7d", "lag_14d", "lag_28d", "lag_365d",
    "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean", "rolling_30d_mean",
    "is_holiday",
]

# Daily weather joined from outputs/weather_daily.csv (historical measurements
# for training, tomorrow's forecast for prediction). LightGBM handles missing
# values natively, so days without weather data simply contribute no signal.
# Rain enters only as daytime_rain_mm (rain during riding hours): with a daily
# total in the mix the model discounts days where rain fell overnight and no
# rider ever saw it (2026-08-19 was a 46% miss for exactly that reason).
WEATHER_FEATURES = [
    "tmax", "tmin", "snow_cm", "wind_max_kmh", "daytime_rain_mm",
]

FEATURES = BASE_FEATURES + WEATHER_FEATURES

ON_HOLIDAYS = holidays.Canada(prov="ON")


def load_weather():
    """Daily weather frame keyed by date, or an empty frame with the expected
    columns if the CSV is missing (features then stay NaN and the model just
    runs weather-blind, same as before weather was added)."""
    if WEATHER_CSV.exists():
        w = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
        # reindex tolerates a stale CSV missing newer columns (they stay NaN)
        return w.reindex(columns=["date"] + WEATHER_FEATURES)
    return pd.DataFrame(columns=["date"] + WEATHER_FEATURES)


def add_weather(featured, weather):
    return featured.merge(weather, on="date", how="left")


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


def build_intraday_profile(hourly_df):
    """Cumulative fraction of a day's ridership reached by each hour, learned from
    complete days. Returns {"weekday": {hr: frac}, "weekend": {...}, "all": {...}}.

    Used to nowcast a still-collecting day: nowcast = partial_so_far / frac[hour].
    Only days that reach DAY_COMPLETE_HOUR are used (a partial day's shape would
    bias the curve). Keyed on the max hour reached, not a row count, because a
    day's intraday hours can have gaps.
    """
    h = hourly_df.copy()
    h["date"] = h["hour"].dt.date
    h["hr"] = h["hour"].dt.hour

    # Per-day hour×total matrix; only keep days whose data reaches end of service.
    last_hour = h.groupby("date")["hr"].max()
    complete_dates = last_hour[last_hour >= DAY_COMPLETE_HOUR].index
    h = h[h["date"].isin(complete_dates)]

    matrix = (
        h.pivot_table(index="date", columns="hr", values="redemptions", aggfunc="sum")
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )
    cum = matrix.cumsum(axis=1)
    day_totals = cum[23].replace(0, np.nan)            # full-day total per day
    fracs = cum.div(day_totals, axis=0).dropna()       # each row → cumulative fraction curve

    is_weekend = pd.to_datetime(fracs.index.to_series()).dt.dayofweek >= 5

    def curve(frame):
        if len(frame) == 0:
            return None
        # Mean fraction per hour, clamped so the divisor is never near zero.
        return {hr: max(float(frame[hr].mean()), 0.02) for hr in range(24)}

    profile = {
        "all": curve(fracs),
        "weekday": curve(fracs[~is_weekend.values]),
        "weekend": curve(fracs[is_weekend.values]),
    }
    # Fall back to the pooled curve when a segment is too sparse to trust.
    for seg in ("weekday", "weekend"):
        seg_count = (~is_weekend).sum() if seg == "weekday" else is_weekend.sum()
        if profile[seg] is None or seg_count < 30:
            profile[seg] = profile["all"]
    return profile


def nowcast_partial_day(daily_df, hourly_df, profile):
    """If the last daily row is a still-collecting partial day, return a copy of
    daily_df with that row's redemptions replaced by a full-day nowcast estimate,
    plus an info dict. Otherwise return daily_df unchanged with is_partial=False.

    The nowcast is used only as the lag basis for forecasting tomorrow — it is
    never treated as the day's actual (that still waits for complete data).
    """
    info = {"is_partial": False, "last_date": None, "maxhr": None,
            "partial": None, "frac": None, "nowcast": None, "seg": None}
    if daily_df.empty or hourly_df.empty:
        return daily_df, info

    last_date = daily_df["date"].max()
    hourly_max_date = hourly_df["hour"].max().normalize()

    # Partial iff the daily and hourly feeds agree on the most recent day AND that
    # day hasn't reached end of service yet.
    if last_date != hourly_max_date:
        return daily_df, info
    maxhr = int(hourly_df[hourly_df["hour"].dt.normalize() == last_date]["hour"].dt.hour.max())
    if maxhr >= DAY_COMPLETE_HOUR:
        return daily_df, info

    partial = int(daily_df.iloc[-1]["redemptions"])
    seg = "weekend" if last_date.dayofweek >= 5 else "weekday"
    frac = profile[seg].get(maxhr, profile["all"][maxhr])
    nowcast = max(round(partial / frac), partial)   # never shrink below what we've already seen

    df_nc = daily_df.copy()
    df_nc.iloc[-1, df_nc.columns.get_loc("redemptions")] = nowcast

    info.update(is_partial=True, last_date=last_date, maxhr=maxhr,
                partial=partial, frac=round(frac, 4), nowcast=int(nowcast), seg=seg)
    return df_nc, info


def train(df, weather):
    featured = add_weather(add_features(df), weather)
    # Drop only on the base features: weather gaps are fine (LightGBM treats
    # NaN as "missing"), but dropping on them would discard training rows.
    featured = featured.dropna(subset=BASE_FEATURES)

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


def forecast_horizon(model, df, weather, residual_std, end_date):
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
        feats = add_weather(add_features(extended), weather)
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


def update_history(df, forecasts, real_tomorrow):
    """Maintain prediction_history.csv so it always matches what the chart shows.

    The forecast for real_tomorrow is the single *provisional* row: it is rewritten
    every run as today's nowcast sharpens, so it stays byte-identical to the chart's
    headline. Once the calendar moves on (real_tomorrow advances past it), the row
    LOCKS at its last value — that frozen value is what gets graded against the
    actual. Backfill rows for earlier skipped days are logged once and locked
    immediately. `locked`: 1 = frozen/gradeable, 0 = still refreshing.
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    actuals = df.set_index("date")["redemptions"]
    real_tomorrow_str = real_tomorrow.strftime("%Y-%m-%d")

    cols = ["date", "day", "predicted", "ci_lower", "ci_upper", "actual", "error_pct", "locked"]
    if HISTORY_CSV.exists():
        history = pd.read_csv(HISTORY_CSV)
    else:
        history = pd.DataFrame(columns=cols)

    # Migrate: legacy rows without `locked` are treated as already locked (settled
    # or being graded) — never retro-refresh them.
    if "locked" not in history.columns:
        history["locked"] = 1
    history["locked"] = history["locked"].fillna(1).astype(int)

    # 1. Lock any provisional row whose day is no longer "tomorrow" — the calendar
    #    has moved on, so its last-written prediction becomes the graded value.
    if not history.empty:
        expired = (history["locked"] != 1) & (history["date"] < real_tomorrow_str)
        history.loc[expired, "locked"] = 1

    # 2. Finalize actuals for locked rows once the day is complete. A day is
    #    complete once a LATER day exists in the data: the feed fills
    #    chronologically, so a newer timestamp can only appear after every earlier
    #    interval. The most recent day may still be collecting, so it is never final.
    latest_date = df["date"].max()
    if not history.empty:
        for i, row in history.iterrows():
            if int(row["locked"]) != 1:
                continue
            existing = row.get("actual")
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

    # 3. Upsert forecast rows.
    #    - real_tomorrow → provisional: overwrite each run, keep locked=0.
    #    - earlier (backfill) days → log once, locked=1, never overwritten.
    history = history.set_index("date", drop=False)
    for fc in forecasts:
        date_str = fc["date"].strftime("%Y-%m-%d")
        is_tomorrow = date_str == real_tomorrow_str
        exists = date_str in history.index
        # Backfill day already logged → leave it (log-once). real_tomorrow is always
        # the future, so its row is always (re)written live, even if a migrated
        # legacy row sat there locked.
        if exists and not is_tomorrow:
            continue
        history.loc[date_str, ["date", "day", "predicted", "ci_lower", "ci_upper", "locked"]] = [
            date_str, day_names[fc["date"].weekday()],
            fc["prediction"], fc["ci_lower"], fc["ci_upper"], 0 if is_tomorrow else 1,
        ]
        if not exists:
            # Unset actual/error_pct → NaN (to_csv writes an empty cell, which the
            # frontend reads as "pending").
            history.loc[date_str, ["actual", "error_pct"]] = [np.nan, np.nan]

    history = history.reset_index(drop=True).sort_values("date").reset_index(drop=True)
    history = history[cols]
    history.to_csv(HISTORY_CSV, index=False)
    return history


def main():
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    hourly = pd.read_csv(HOURLY_CSV, parse_dates=["hour"])
    weather = load_weather()
    profile = build_intraday_profile(hourly)
    df_nc, nc = nowcast_partial_day(df, hourly, profile)

    # Train on complete days only — never let today's partial total pollute the
    # validation split / residual_std (which sets the CI width).
    train_df = df.iloc[:-1] if nc["is_partial"] else df
    model, mae, r2, residual_std = train(train_df, weather)

    # "Tomorrow" is the real-world next day in Toronto time. Forecast on df_nc so
    # today's lag is the nowcast (a sane full-day estimate) rather than its partial
    # sum; the horizon reaches real tomorrow in both regimes (1 step when today's
    # partial data is present, recursive backfill when the feed lags).
    today = pd.Timestamp(datetime.now(LOCAL_TZ).date())
    real_tomorrow = today + timedelta(days=1)
    src = df_nc if nc["is_partial"] else df
    forecasts = forecast_horizon(model, src, weather, residual_std, real_tomorrow)

    headline = next((f for f in forecasts if f["date"] == real_tomorrow), forecasts[-1])
    tomorrow = headline["date"]

    context = build_context(df_nc, tomorrow)
    # Flag today's bar as a nowcast estimate so the chart can style it distinctly
    # (it's an estimate scaled from partial data, not a measured actual).
    if nc["is_partial"]:
        nc_date = nc["last_date"].strftime("%Y-%m-%d")
        for entry in context["last_7d"]:
            if entry["date"] == nc_date:
                entry["nowcast"] = True
                entry["observed_pct"] = round(nc["frac"] * 100)
                break

    update_history(df, forecasts, real_tomorrow)

    if nc["is_partial"]:
        print(f"Nowcast: {nc['last_date'].date()} partial {nc['partial']:,} through "
              f"hr {nc['maxhr']} ({nc['seg']}, {nc['frac']:.0%}) -> {nc['nowcast']:,} full-day")

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
