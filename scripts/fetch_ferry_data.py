"""
Fetch Toronto Island Ferry ticket counts from Toronto Open Data and save as CSV.
"""
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"

# Get package metadata
url = base_url + "/api/3/action/package_show"
params = {"id": "toronto-island-ferry-ticket-counts"}
package = requests.get(url, params=params, timeout=30).json()

# Find the active datastore resource and download it
for resource in package["result"]["resources"]:
    if resource["datastore_active"]:
        dump_url = base_url + "/datastore/dump/" + resource["id"]
        response = requests.get(dump_url, timeout=60)
        df = pd.read_csv(StringIO(response.text))

        # Clean up
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Save
        out_path = Path(__file__).parent.parent / "outputs" / "ferry_ticket_counts.csv"
        df.to_csv(out_path, index=False)

        print(f"Saved {len(df)} records to {out_path}")
        print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        break
