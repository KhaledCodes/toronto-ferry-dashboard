"""
Fetch daily Toronto weather (history + tomorrow's forecast) from Open-Meteo
and save as outputs/weather_daily.csv for the prediction model.

Two endpoints are combined because neither alone covers the full range:
- the archive API has measured history but lags a few days behind today
- the forecast API covers the recent past and the next few days

Measured (archive) values win when both have a date; is_forecast marks rows
that came from the forecast API so they can be told apart later.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# Jack Layton Ferry Terminal
LAT, LON = 43.62, -79.38
START_DATE = "2015-05-01"

DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "rain_sum,snowfall_sum,wind_speed_10m_max"
)
COLUMN_NAMES = {
    "time": "date",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "precip_mm",
    "rain_sum": "rain_mm",
    "snowfall_sum": "snow_cm",
    "wind_speed_10m_max": "wind_max_kmh",
}

OUT_PATH = Path(__file__).parent.parent / "outputs" / "weather_daily.csv"
HOURLY_OUT_PATH = Path(__file__).parent.parent / "outputs" / "weather_hourly.csv"

# A daily precipitation total can't tell overnight rain from rain that actually
# deters riders (2026-08-19: 7.7mm total but ~90% fell 3-7am, and ridership was
# normal). daytime_rain_mm sums only the hours when people ride ferries.
DAYTIME_START_HOUR, DAYTIME_END_HOUR = 9, 21


def fetch_daily(url, extra_params):
    """Returns (daily_df, hourly_df) for the requested range."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": DAILY_VARS,
        "hourly": "precipitation,temperature_2m",
        "timezone": "America/Toronto",
        **extra_params,
    }
    payload = requests.get(url, params=params, timeout=60).json()
    if "daily" not in payload:
        raise RuntimeError(f"Unexpected Open-Meteo response: {payload}")
    df = pd.DataFrame(payload["daily"]).rename(columns=COLUMN_NAMES)
    df["date"] = pd.to_datetime(df["date"])

    hourly = pd.DataFrame(payload["hourly"])
    hourly["date"] = pd.to_datetime(hourly["time"].str[:10])
    hourly["hour"] = hourly["time"].str[11:13].astype(int)
    daytime = (
        hourly[hourly["hour"].between(DAYTIME_START_HOUR, DAYTIME_END_HOUR)]
        .groupby("date")["precipitation"].sum()
        .rename("daytime_rain_mm")
    )
    return df.merge(daytime, on="date", how="left"), hourly


def main():
    archive, _ = fetch_daily(
        "https://archive-api.open-meteo.com/v1/archive",
        {"start_date": START_DATE, "end_date": date.today().isoformat()},
    )
    # The archive's trailing rows can be placeholders with no readings yet.
    archive = archive.dropna(subset=["tmax"])
    archive["is_forecast"] = 0

    forecast, forecast_hourly = fetch_daily(
        "https://api.open-meteo.com/v1/forecast",
        {"past_days": 14, "forecast_days": 3},
    )
    forecast["is_forecast"] = 1

    # Recent hourly temperatures for the dashboard's 1-day (hourly) chart view.
    hourly_out = forecast_hourly.rename(columns={"temperature_2m": "temp_c"})
    hourly_out[["time", "temp_c"]].to_csv(HOURLY_OUT_PATH, index=False)
    print(f"Saved {len(hourly_out)} hourly temperature rows to {HOURLY_OUT_PATH}")

    # Archive rows win; forecast fills the gap between archive and tomorrow.
    combined = (
        pd.concat([archive, forecast[~forecast["date"].isin(archive["date"])]])
        .sort_values("date")
        .reset_index(drop=True)
    )

    combined.to_csv(OUT_PATH, index=False)
    n_forecast = int(combined["is_forecast"].sum())
    print(f"Saved {len(combined)} days of weather to {OUT_PATH}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}"
          f" ({n_forecast} forecast rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Weather is an enhancement — a fetch outage must not block the
        # ridership pipeline. Keep the previous CSV and carry on, but fail
        # loudly if there is no CSV at all to fall back to.
        if OUT_PATH.exists():
            print(f"WARNING: weather fetch failed ({exc}); keeping existing {OUT_PATH}")
            sys.exit(0)
        raise
