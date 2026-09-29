# Toronto Island Ferry Dashboard

![Hourly Update](https://github.com/KhaledCodes/toronto-ferry-dashboard/actions/workflows/hourly_update.yml/badge.svg)

A live dashboard tracking Toronto Island Ferry ridership, built with D3.js and served via GitHub Pages. Data comes from the Toronto Open Data API as 15-minute interval records. The feed updates irregularly and can stall for stretches, so a GitHub Actions pipeline polls it hourly to pick up new data and backfills as they land.

## Live Demo

**[View the Dashboard](https://khaledcodes.github.io/toronto-ferry-dashboard/)**

## Features

- **Live Data**: Polled hourly from the Toronto Open Data 15-minute interval feed
- **Multiple Time Scales**: 1D (hourly), 7D, 14D, 30D, 90D, 1Y, and all-time views
- **Interactive Charts**: Hover tooltips, responsive D3.js area chart
- **KPI Summary**: Latest day, 7-day total, peak day, and year-over-year comparison

## Data Source

- **Ferry Ticket Counts**: [Toronto Open Data](https://open.toronto.ca/dataset/toronto-island-ferry-ticket-counts/) - 15-minute interval ticket redemptions since 2015

## How It Works

1. **GitHub Actions** runs hourly (and on demand)
2. `fetch_ferry_data.py` pulls the latest ticket data from the Toronto Open Data API
3. `aggregate_daily.py` aggregates to hourly and daily totals
4. Updated CSVs are committed back to the repo
5. GitHub Pages serves `index.html`, which loads the CSVs with D3.js

### Freshness safeguards

Downloads bypass shared caches and are checked against the live datastore's
latest timestamp. If the primary dump is stale or fails, the fetcher tries the
City's alternate CSV download. Data more than 48 hours old, behind the live
datastore, or missing the existing history is rejected before replacing any files.
If both sources fail, the workflow fails and its existing issue notification fires.

The dashboard flags delayed observations and only displays a forecast whose date
is tomorrow in Toronto. The prediction script also suppresses forecasts when
observations are more than 48 hours old.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
node tests/test_dashboard_freshness.cjs
```

## Project Structure

```
toronto-ferry-dashboard/
├── index.html                     # Dashboard (D3.js)
├── scripts/
│   ├── fetch_ferry_data.py        # Fetches raw data from Toronto Open Data
│   └── aggregate_daily.py         # Aggregates to daily + hourly CSVs
├── outputs/
│   ├── ferry_ticket_counts.csv    # Raw 15-min ticket data
│   ├── daily_totals.csv           # Daily redemption totals
│   └── hourly_totals.csv          # Hourly redemption totals
├── .github/workflows/
│   └── hourly_update.yml          # GitHub Actions hourly pipeline
├── archive/                       # Previous Streamlit + ML pipeline
└── requirements.txt
```

## Local Development

```bash
git clone https://github.com/KhaledCodes/toronto-ferry-dashboard.git
cd toronto-ferry-dashboard

# Fetch fresh data
pip install requests pandas
python scripts/fetch_ferry_data.py
python scripts/aggregate_daily.py

# Serve locally
python -m http.server 8090
# Open http://localhost:8090
```

## License

MIT License - feel free to use and modify for your own projects.
