# Toronto Island Ferry Dashboard

A real-time analytics dashboard for Toronto Island Ferry ridership data with 7-day weather-based forecasts and long-term predictions.

## Live Demo

🚀 **[View the Dashboard](https://your-app-name.streamlit.app)** *(update after deployment)*

## Features

- **Real-time Analytics**: View current and historical ridership patterns
- **7-Day Forecasts**: Weather-based predictions using LightGBM models
- **Long-Term Outlook**: 365-day forecasts using historical weather patterns
- **World Cup 2026**: Special tourism features for FIFA World Cup Toronto matches
- **Interactive Charts**: Filter by time granularity, compare redemptions vs sales

## Data Sources

- **Ferry Data**: [Toronto Open Data](https://open.toronto.ca/dataset/toronto-island-ferry-ticket-counts/)
- **Weather Data**: [Meteostat](https://meteostat.net/) & [Open-Meteo](https://open-meteo.com/)

## Deployment

### Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from `dashboard/Home.py`

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/toronto-ferry-dashboard.git
cd toronto-ferry-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/Home.py
```

## Automatic Daily Updates

This repository uses GitHub Actions to automatically update data daily at 8:00 AM UTC:

1. Fetches latest ferry ticket data from Toronto Open Data API
2. Updates weather data and forecasts
3. Regenerates 7-day and long-term predictions
4. Commits updated CSVs back to the repository

The workflow can also be triggered manually from the Actions tab.

## Project Structure

```
toronto-ferry-dashboard/
├── .github/
│   └── workflows/
│       └── daily_update.yml    # GitHub Actions for daily updates
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── dashboard/
│   ├── Home.py                 # Main dashboard page
│   └── pages/
│       ├── 1_Historical.py     # Historical trends
│       ├── 2_Forecast.py       # 7-day & long-term forecasts
│       ├── 3_Weather.py        # Weather analysis
│       └── 4_Calendar.py       # Calendar view
├── models/
│   ├── redemption_model.pkl    # Trained LightGBM model (redemptions)
│   ├── sales_model.pkl         # Trained LightGBM model (sales)
│   ├── *_no_weather.pkl        # Models for long-term forecasts
│   └── model_metadata.json     # Model training metadata
├── outputs/
│   ├── hourly_data.csv         # Historical hourly data
│   ├── forecasts.csv           # 7-day hourly forecasts
│   ├── daily_forecasts.csv     # 7-day daily summary
│   └── long_term_*.csv         # Long-term forecast files
├── scripts/
│   ├── daily_update.py         # Daily data update script
│   ├── generate_forecasts.py   # Forecast generation
│   └── train_models.py         # Model training
├── src/
│   ├── config.py               # Configuration settings
│   ├── feature_engineering.py  # Feature creation
│   ├── model_inference.py      # Prediction logic
│   ├── tourism_features.py     # Tourism & World Cup features
│   └── weather_climatology.py  # Historical weather patterns
├── requirements.txt            # Python dependencies
└── README.md
```

## Model Information

- **Algorithm**: LightGBM (Gradient Boosting)
- **Features**: 23+ features including weather, time, holidays, tourism
- **Target Variables**: Ticket redemptions and sales (separate models)
- **Training Data**: 2015-present (~10 years of hourly data)

## License

MIT License - feel free to use and modify for your own projects.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
