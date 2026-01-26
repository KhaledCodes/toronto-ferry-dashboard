"""
Toronto Island Ferry Dashboard
Main Streamlit application entry point.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Home - Toronto Ferry Dashboard",
    page_icon=":ferry:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
HOURLY_DATA_PATH = PROJECT_ROOT / "outputs" / "hourly_data.csv"
FORECASTS_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
DAILY_FORECASTS_PATH = PROJECT_ROOT / "outputs" / "daily_forecasts.csv"


@st.cache_data(ttl=3600)
def load_hourly_data():
    """Load hourly historical data."""
    df = pd.read_csv(HOURLY_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


@st.cache_data(ttl=3600)
def load_forecasts():
    """Load forecast data."""
    if FORECASTS_PATH.exists():
        df = pd.read_csv(FORECASTS_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_daily_forecasts():
    """Load daily forecast summary."""
    if DAILY_FORECASTS_PATH.exists():
        df = pd.read_csv(DAILY_FORECASTS_PATH)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    return None


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This dashboard visualizes **Toronto Island Ferry**
        ridership data and provides **7-day forecasts**
        based on weather predictions.

        **Data Sources:**
        - Toronto Open Data (ferry tickets)
        - Meteostat & Open-Meteo (weather)

        **Model:** LightGBM with 23+ features
        """)

        st.markdown("---")
        st.markdown("*Data updates daily at 8:00 UTC*")

    # Main content
    st.markdown('<p class="main-header">Toronto Island Ferry Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time ridership analytics and 7-day forecasts</p>', unsafe_allow_html=True)

    # Load data
    hourly_df = load_hourly_data()
    forecasts_df = load_forecasts()
    daily_forecasts = load_daily_forecasts()

    # KPI metrics row
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)

    # Recent data (last available day)
    recent_df = hourly_df[hourly_df['date'] >= (hourly_df['date'].max() - timedelta(days=7))]
    last_day = hourly_df[hourly_df['date'] == hourly_df['date'].max()]

    with col1:
        last_day_total = int(last_day['redemption_count'].sum()) if len(last_day) > 0 else 0
        st.metric(
            label="Last Day Total",
            value=f"{last_day_total:,}",
            help="Total redemptions on most recent data day"
        )

    with col2:
        week_avg = int(recent_df.groupby('date')['redemption_count'].sum().mean()) if len(recent_df) > 0 else 0
        st.metric(
            label="7-Day Avg",
            value=f"{week_avg:,}",
            help="Average daily redemptions over last 7 days"
        )

    with col3:
        if daily_forecasts is not None and len(daily_forecasts) > 0:
            forecast_total = int(daily_forecasts['total_redemptions'].sum())
            st.metric(
                label="7-Day Forecast",
                value=f"{forecast_total:,}",
                help="Total predicted redemptions for next 7 days"
            )
        else:
            st.metric(label="7-Day Forecast", value="N/A")

    with col4:
        total_records = len(hourly_df)
        st.metric(
            label="Total Records",
            value=f"{total_records:,}",
            help="Total hourly records in database"
        )

    st.markdown("---")

    # Two column layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Recent Ridership Trend")

        # Last 30 days daily aggregation
        recent_30 = hourly_df[hourly_df['timestamp'] >= (hourly_df['timestamp'].max() - timedelta(days=30))]
        daily_recent = recent_30.groupby('date').agg({
            'redemption_count': 'sum',
            'sales_count': 'sum'
        }).reset_index()

        fig = px.line(
            daily_recent,
            x='date',
            y='redemption_count',
            title='Daily Redemptions (Last 30 Days)',
            labels={'date': 'Date', 'redemption_count': 'Redemptions'}
        )
        fig.update_layout(
            hovermode='x unified',
            showlegend=False
        )
        fig.update_traces(line_color='#1E88E5', line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("7-Day Forecast")

        if daily_forecasts is not None and len(daily_forecasts) > 0:
            # Simple bar chart of forecast
            fig = px.bar(
                daily_forecasts,
                x='date',
                y='total_redemptions',
                title='Predicted Daily Redemptions',
                labels={'date': 'Date', 'total_redemptions': 'Predicted'}
            )
            fig.update_traces(marker_color='#43A047')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forecasts available. Run forecast generation script.")

    # Quick stats section
    st.markdown("---")
    st.subheader("Quick Stats")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Coverage**")
        min_date = hourly_df['timestamp'].min().strftime('%Y-%m-%d')
        max_date = hourly_df['timestamp'].max().strftime('%Y-%m-%d')
        st.write(f"From: {min_date}")
        st.write(f"To: {max_date}")

    with col2:
        st.markdown("**Peak Season (Jul-Aug)**")
        summer = hourly_df[hourly_df['month'].isin([7, 8])]
        if len(summer) > 0:
            summer_avg = int(summer.groupby('date')['redemption_count'].sum().mean())
            st.write(f"Avg Daily: {summer_avg:,}")

    with col3:
        st.markdown("**Off Season (Dec-Feb)**")
        winter = hourly_df[hourly_df['month'].isin([12, 1, 2])]
        if len(winter) > 0:
            winter_avg = int(winter.groupby('date')['redemption_count'].sum().mean())
            st.write(f"Avg Daily: {winter_avg:,}")


if __name__ == "__main__":
    main()
