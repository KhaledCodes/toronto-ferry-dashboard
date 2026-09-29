"""Fetch and validate Toronto ferry data before replacing the last good CSV."""
from datetime import datetime
from io import StringIO
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "ferry_ticket_counts.csv"
LOCAL_TZ = ZoneInfo("America/Toronto")
MAX_DATA_AGE = pd.Timedelta(hours=48)
COLUMNS = ["Timestamp", "Redemption Count", "Sales Count"]


def get_fresh(url, **params):
    # A successful cached dump can be weeks behind the live datastore. Give each
    # request a unique cache key, including retries and metadata requests.
    if url.endswith("/datastore_search"):
        # CKAN rejects unknown query fields on search. POST bypasses shared GET
        # caches without adding a field to its strict search schema.
        response = requests.post(url, json=params, headers={"Cache-Control": "no-cache"}, timeout=90)
    else:
        response = requests.get(
            url, params={**params, "_refresh": uuid4().hex},
            headers={"Cache-Control": "no-cache"}, timeout=90,
        )
    response.raise_for_status()
    return response


def validate_csv(text, now, expected_latest=None, previous=None):
    df = pd.read_csv(StringIO(text))
    if df.empty or not set(COLUMNS).issubset(df.columns):
        raise ValueError("Empty ferry data or missing required columns")
    df = df[COLUMNS].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
    for col in COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.isna().any().any() or (df[COLUMNS[1:]] < 0).any().any():
        raise ValueError("Ferry data contains missing or negative values")
    latest = df["Timestamp"].max()
    if now - latest > MAX_DATA_AGE:
        raise ValueError(f"Stale ferry data: latest record is {latest}")
    if latest > now + pd.Timedelta(hours=1):
        raise ValueError(f"Ferry data has a future timestamp: {latest}")
    if expected_latest is not None and latest < expected_latest:
        raise ValueError(f"Download ends at {latest}; live datastore reaches {expected_latest}")
    if previous is not None:
        if latest < previous.max() or df["Timestamp"].min() > previous.min():
            raise ValueError("Download would discard existing ferry history")
    return df.sort_values("Timestamp").reset_index(drop=True)


def fetch_data(out_path=OUT_PATH, now=None):
    now = pd.Timestamp(now if now is not None else datetime.now(LOCAL_TZ).replace(tzinfo=None))
    package = get_fresh(
        BASE_URL + "/api/3/action/package_show", id="toronto-island-ferry-ticket-counts",
    ).json()
    if not package.get("success"):
        raise RuntimeError("Toronto package metadata request failed")
    resources = package["result"]["resources"]
    active = next((r for r in resources if r.get("datastore_active")), None)
    if active is None:
        raise RuntimeError("No active ferry datastore found")

    expected_latest = None
    try:
        probe = get_fresh(
            BASE_URL + "/api/3/action/datastore_search",
            resource_id=active["id"], limit=1, sort="Timestamp desc",
        ).json()
        expected_latest = pd.Timestamp(probe["result"]["records"][0]["Timestamp"])
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        print(f"WARNING: live timestamp check failed ({exc}); enforcing age and history checks")

    previous = None
    if out_path.exists():
        previous = pd.read_csv(out_path, usecols=["Timestamp"], parse_dates=["Timestamp"])["Timestamp"]
    dump_url = BASE_URL + "/datastore/dump/" + active["id"]
    alternatives = [r["url"] for r in resources if not r.get("datastore_active")
                    and (r.get("format", "").lower() == "csv" or r.get("url", "").lower().endswith(".csv"))]
    errors = []
    for url in dict.fromkeys([dump_url, *alternatives]):
        try:
            df = validate_csv(get_fresh(url).text, now, expected_latest, previous)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            print(f"WARNING: rejected ferry source: {errors[-1]}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = out_path.with_suffix(".csv.tmp")
        df.to_csv(temporary, index=False)
        temporary.replace(out_path)
        print(f"Saved {len(df)} records to {out_path}")
        print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"Source: {url}")
        return df
    raise RuntimeError("No fresh ferry source; preserved existing data. " + "; ".join(errors))


if __name__ == "__main__":
    fetch_data()
