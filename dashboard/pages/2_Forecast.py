"""
Forecast Page
View 7-day ridership predictions and long-term outlook.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Forecast", page_icon=":crystal_ball:", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent
HOURLY_DATA_PATH = PROJECT_ROOT / "outputs" / "hourly_data.csv"
FORECASTS_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
DAILY_FORECASTS_PATH = PROJECT_ROOT / "outputs" / "daily_forecasts.csv"
LONG_TERM_FORECASTS_PATH = PROJECT_ROOT / "outputs" / "long_term_forecasts.csv"
LONG_TERM_DAILY_PATH = PROJECT_ROOT / "outputs" / "long_term_daily_forecasts.csv"
LONG_TERM_MONTHLY_PATH = PROJECT_ROOT / "outputs" / "long_term_monthly_forecasts.csv"


@st.cache_data(ttl=3600)
def load_historical():
    df = pd.read_csv(HOURLY_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=3600)
def load_forecasts():
    if FORECASTS_PATH.exists():
        df = pd.read_csv(FORECASTS_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_daily_forecasts():
    if DAILY_FORECASTS_PATH.exists():
        df = pd.read_csv(DAILY_FORECASTS_PATH)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_long_term_forecasts():
    if LONG_TERM_FORECASTS_PATH.exists():
        df = pd.read_csv(LONG_TERM_FORECASTS_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_long_term_daily():
    if LONG_TERM_DAILY_PATH.exists():
        df = pd.read_csv(LONG_TERM_DAILY_PATH)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_long_term_monthly():
    if LONG_TERM_MONTHLY_PATH.exists():
        df = pd.read_csv(LONG_TERM_MONTHLY_PATH)
        df['year_month'] = pd.to_datetime(df['year_month'])
        return df
    return None


def aggregate_data(df, granularity, timestamp_col='timestamp', value_col='redemption_count'):
    """Aggregate data based on selected time granularity."""
    df = df.copy()
    col = value_col

    if granularity == 'Hourly':
        return df[[timestamp_col, col]].rename(columns={col: 'value'})

    df['period'] = df[timestamp_col]

    if granularity == 'Daily':
        df['period'] = df[timestamp_col].dt.date
    elif granularity == 'Weekly':
        df['period'] = df[timestamp_col].dt.to_period('W').apply(lambda x: x.start_time)
    elif granularity == 'Monthly':
        df['period'] = df[timestamp_col].dt.to_period('M').apply(lambda x: x.start_time)
    elif granularity == 'Yearly':
        df['period'] = df[timestamp_col].dt.to_period('Y').apply(lambda x: x.start_time)

    agg_df = df.groupby('period')[col].sum().reset_index()
    agg_df.columns = ['period', 'value']
    agg_df['period'] = pd.to_datetime(agg_df['period'])
    return agg_df


def render_7day_forecast(forecasts, daily_forecasts, historical, long_term_forecasts):
    """Render the 7-day forecast tab content."""
    if forecasts is None or len(forecasts) == 0:
        st.warning("No 7-day forecasts available. Please run the forecast generation script.")
        st.code("python scripts/generate_forecasts.py", language="bash")
        return

    # Forecast summary metrics
    st.subheader("Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_redemptions = int(forecasts['predicted_redemption_count'].sum())
        st.metric("Total Redemptions", f"{total_redemptions:,}")

    with col2:
        total_sales = int(forecasts['predicted_sales_count'].sum())
        st.metric("Total Sales", f"{total_sales:,}")

    with col3:
        avg_temp = forecasts['temp'].mean()
        st.metric("Avg Temperature", f"{avg_temp:.1f}C")

    with col4:
        forecast_period = f"{forecasts['timestamp'].min().strftime('%b %d')} - {forecasts['timestamp'].max().strftime('%b %d')}"
        st.metric("Forecast Period", forecast_period)

    st.markdown("---")

    # Daily forecast table
    st.subheader("Daily Forecast")

    if daily_forecasts is not None:
        display_df = daily_forecasts[['date', 'total_redemptions', 'total_sales']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d (%a)')
        display_df.columns = ['Date', 'Redemptions', 'Sales']

        st.dataframe(
            display_df.style.format({
                'Redemptions': '{:,.0f}',
                'Sales': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Predictions chart with time granularity selector
    st.subheader("Predictions")

    col_gran, col_metric = st.columns([2, 1])
    with col_gran:
        granularity = st.selectbox(
            "Select time granularity",
            options=['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'],
            index=0,
            key='7day_granularity'
        )
    with col_metric:
        metric_type = st.radio(
            "Metric",
            options=['Redemptions', 'Sales'],
            horizontal=True,
            key='7day_metric'
        )

    # Set column names based on metric type
    if metric_type == 'Redemptions':
        hist_col = 'redemption_count'
        forecast_col = 'predicted_redemption_count'
    else:
        hist_col = 'sales_count'
        forecast_col = 'predicted_sales_count'

    last_hist_date = historical['timestamp'].max()
    if granularity == 'Hourly':
        lookback = timedelta(days=3)
    elif granularity == 'Daily':
        lookback = timedelta(days=30)
    elif granularity == 'Weekly':
        lookback = timedelta(days=90)
    elif granularity == 'Monthly':
        lookback = timedelta(days=365)
    else:
        lookback = timedelta(days=365 * 10)

    recent_hist = historical[historical['timestamp'] >= last_hist_date - lookback]

    hist_agg = aggregate_data(recent_hist, granularity, 'timestamp', hist_col)
    forecast_agg = aggregate_data(forecasts, granularity, 'timestamp', forecast_col)

    fig = go.Figure()

    x_col = 'period' if granularity != 'Hourly' else 'timestamp'
    fig.add_trace(go.Bar(
        x=hist_agg[x_col if x_col in hist_agg.columns else 'period'],
        y=hist_agg['value'],
        name='Historical',
        marker_color='#1E88E5'
    ))

    fig.add_trace(go.Bar(
        x=forecast_agg[x_col if x_col in forecast_agg.columns else 'period'],
        y=forecast_agg['value'],
        name='7-Day Forecast',
        marker_color='#E91E63'
    ))

    # Add long-term forecast if available
    if long_term_forecasts is not None and len(long_term_forecasts) > 0:
        long_term_agg = aggregate_data(long_term_forecasts, granularity, 'timestamp', forecast_col)
        fig.add_trace(go.Bar(
            x=long_term_agg[x_col if x_col in long_term_agg.columns else 'period'],
            y=long_term_agg['value'],
            name='Long-Term Forecast',
            marker_color='#9C27B0'
        ))

    title = f'{granularity} {metric_type}: Historical vs Forecast'
    y_label = f'Total {metric_type}' if granularity != 'Hourly' else metric_type

    fig.update_layout(
        title=title,
        xaxis_title='Date/Time',
        yaxis_title=y_label,
        hovermode='x unified',
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weekly bar chart (Historical vs Forecast comparison)
    st.markdown("---")
    st.subheader("Weekly Comparison")

    # Always aggregate to weekly for this bar chart
    hist_weekly = aggregate_data(historical, 'Weekly', 'timestamp', 'redemption_count')
    forecast_weekly = aggregate_data(forecasts, 'Weekly', 'timestamp', 'predicted_redemption_count')

    # Get last 8 weeks of historical + forecast weeks
    hist_weekly = hist_weekly.tail(8)

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=hist_weekly['period'],
        y=hist_weekly['value'],
        name='Historical',
        marker_color='#1E88E5'
    ))

    fig_bar.add_trace(go.Bar(
        x=forecast_weekly['period'],
        y=forecast_weekly['value'],
        name='7-Day Forecast',
        marker_color='#E91E63'
    ))

    # Add long-term forecast if available
    if long_term_forecasts is not None and len(long_term_forecasts) > 0:
        long_term_weekly = aggregate_data(long_term_forecasts, 'Weekly', 'timestamp', 'predicted_redemption_count')
        # Show next 12 weeks of long-term forecast
        long_term_weekly = long_term_weekly.head(12)
        fig_bar.add_trace(go.Bar(
            x=long_term_weekly['period'],
            y=long_term_weekly['value'],
            name='Long-Term Forecast',
            marker_color='#9C27B0'
        ))

    fig_bar.update_layout(
        title='Weekly Redemptions: Historical vs Forecast',
        xaxis_title='Week Starting',
        yaxis_title='Total Redemptions',
        hovermode='x unified',
        height=450,
        barmode='group',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Weather impact section
    st.markdown("---")
    st.subheader("Weather Impact")

    col1, col2 = st.columns(2)

    with col1:
        fig_temp = px.scatter(
            forecasts,
            x='temp',
            y='predicted_redemption_count',
            color='hour_of_day',
            title='Temperature vs Predicted Ridership',
            labels={
                'temp': 'Temperature (C)',
                'predicted_redemption_count': 'Predicted Redemptions',
                'hour_of_day': 'Hour'
            }
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        if daily_forecasts is not None:
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Bar(
                x=daily_forecasts['date'],
                y=daily_forecasts['total_redemptions'],
                name='Redemptions',
                marker_color='#1E88E5'
            ))
            fig_daily.add_trace(go.Bar(
                x=daily_forecasts['date'],
                y=daily_forecasts['total_sales'],
                name='Sales',
                marker_color='#43A047'
            ))
            fig_daily.update_layout(
                title='Daily Forecast: Redemptions vs Sales',
                xaxis_title='Date',
                yaxis_title='Count',
                barmode='group',
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_daily, use_container_width=True)


def render_long_term_outlook(long_term_daily, long_term_monthly, historical):
    """Render the Long-Term Outlook tab content."""
    if long_term_monthly is None:
        st.warning("No long-term forecasts available. Please run the forecast generation script with --long-term flag.")
        st.code("python scripts/generate_forecasts.py --long-term", language="bash")
        st.info("Note: You must first train the no-weather models:")
        st.code("python scripts/train_models.py --no-weather", language="bash")
        return

    st.subheader("12-Month Ridership Outlook")
    st.markdown("Based on historical weather patterns and seasonal trends")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_year_redemptions = int(long_term_monthly['total_redemptions'].sum())
    total_year_sales = int(long_term_monthly['total_sales'].sum())
    peak_month = long_term_monthly.loc[long_term_monthly['total_redemptions'].idxmax()]
    avg_monthly = int(long_term_monthly['total_redemptions'].mean())

    with col1:
        st.metric("Annual Redemptions", f"{total_year_redemptions:,}")

    with col2:
        st.metric("Annual Sales", f"{total_year_sales:,}")

    with col3:
        st.metric("Peak Month", peak_month['year_month'].strftime('%b %Y'))

    with col4:
        st.metric("Avg Monthly", f"{avg_monthly:,}")

    st.markdown("---")

    # Monthly forecast chart with confidence bands
    st.subheader("Monthly Forecast")

    fig = go.Figure()

    # Confidence band (lower to upper)
    fig.add_trace(go.Scatter(
        x=pd.concat([long_term_monthly['year_month'], long_term_monthly['year_month'][::-1]]),
        y=pd.concat([long_term_monthly['redemption_upper'], long_term_monthly['redemption_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band',
        showlegend=True
    ))

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=long_term_monthly['year_month'],
        y=long_term_monthly['total_redemptions'],
        mode='lines+markers',
        name='Predicted Redemptions',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=8)
    ))

    # Add historical comparison if available
    if historical is not None and len(historical) > 0:
        hist_monthly = historical.copy()
        hist_monthly['year_month'] = hist_monthly['timestamp'].dt.to_period('M').dt.to_timestamp()
        hist_agg = hist_monthly.groupby('year_month')['redemption_count'].sum().reset_index()

        # Get same months from previous year for comparison
        forecast_months = set(long_term_monthly['year_month'].dt.month)
        prev_year = long_term_monthly['year_month'].min().year - 1
        hist_same_months = hist_agg[
            (hist_agg['year_month'].dt.year == prev_year) &
            (hist_agg['year_month'].dt.month.isin(forecast_months))
        ].copy()

        if len(hist_same_months) > 0:
            # Shift dates to align with forecast year for comparison
            hist_same_months['year_month'] = hist_same_months['year_month'] + pd.DateOffset(years=1)
            fig.add_trace(go.Scatter(
                x=hist_same_months['year_month'],
                y=hist_same_months['redemption_count'],
                mode='lines',
                name=f'Previous Year ({prev_year})',
                line=dict(color='#9E9E9E', width=2, dash='dot')
            ))

    fig.update_layout(
        title='Monthly Redemptions Forecast with Confidence Band',
        xaxis_title='Month',
        yaxis_title='Total Redemptions',
        hovermode='x unified',
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly summary table
    st.markdown("---")
    st.subheader("Monthly Summary")

    display_monthly = long_term_monthly.copy()
    display_monthly['year_month'] = display_monthly['year_month'].dt.strftime('%b %Y')
    display_monthly = display_monthly[['year_month', 'total_redemptions', 'total_sales', 'forecast_confidence']]
    display_monthly.columns = ['Month', 'Redemptions', 'Sales', 'Confidence']

    # Color code confidence
    def highlight_confidence(val):
        if val == 'high':
            return 'background-color: #C8E6C9'
        elif val == 'medium':
            return 'background-color: #FFF9C4'
        else:
            return 'background-color: #FFCDD2'

    st.dataframe(
        display_monthly.style.format({
            'Redemptions': '{:,.0f}',
            'Sales': '{:,.0f}'
        }).map(highlight_confidence, subset=['Confidence']),
        use_container_width=True,
        hide_index=True
    )

    # Seasonal analysis
    st.markdown("---")
    st.subheader("Seasonal Analysis")

    # Add season to monthly data
    monthly_with_season = long_term_monthly.copy()
    monthly_with_season['month'] = monthly_with_season['year_month'].dt.month
    monthly_with_season['season'] = monthly_with_season['month'].apply(
        lambda m: 'Winter' if m in [12, 1, 2] else
                  'Spring' if m in [3, 4, 5] else
                  'Summer' if m in [6, 7, 8] else 'Fall'
    )

    seasonal_summary = monthly_with_season.groupby('season').agg({
        'total_redemptions': 'sum',
        'total_sales': 'sum'
    }).reset_index()

    # Reorder seasons
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_summary['season'] = pd.Categorical(seasonal_summary['season'], categories=season_order, ordered=True)
    seasonal_summary = seasonal_summary.sort_values('season')

    col1, col2 = st.columns(2)

    with col1:
        fig_season = px.bar(
            seasonal_summary,
            x='season',
            y='total_redemptions',
            title='Predicted Redemptions by Season',
            color='season',
            color_discrete_map={
                'Spring': '#66BB6A',
                'Summer': '#FFA726',
                'Fall': '#8D6E63',
                'Winter': '#42A5F5'
            }
        )
        fig_season.update_layout(showlegend=False)
        st.plotly_chart(fig_season, use_container_width=True)

    with col2:
        # Pie chart of seasonal distribution
        fig_pie = px.pie(
            seasonal_summary,
            values='total_redemptions',
            names='season',
            title='Seasonal Distribution',
            color='season',
            color_discrete_map={
                'Spring': '#66BB6A',
                'Summer': '#FFA726',
                'Fall': '#8D6E63',
                'Winter': '#42A5F5'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.info(f"""
        **Peak Season**: {seasonal_summary.loc[seasonal_summary['total_redemptions'].idxmax(), 'season']}

        **Expected Peak Month**: {peak_month['year_month'].strftime('%B %Y')}

        **Peak Month Redemptions**: {int(peak_month['total_redemptions']):,}
        """)

    with insights_col2:
        # Calculate year-over-year if we have historical data
        yoy_msg = "Compare with historical data for year-over-year analysis"
        if historical is not None:
            hist_total = historical['redemption_count'].sum()
            if hist_total > 0:
                yoy_change = ((total_year_redemptions - hist_total / (len(historical) / 8760) * 1) / (hist_total / (len(historical) / 8760))) * 100
                yoy_msg = f"Estimated annual trend based on historical patterns"

        st.info(f"""
        **Annual Total**: {total_year_redemptions:,} redemptions

        **Monthly Average**: {avg_monthly:,} redemptions

        {yoy_msg}
        """)


def main():
    st.title(":crystal_ball: Ridership Forecast")

    # Load all data
    forecasts = load_forecasts()
    daily_forecasts = load_daily_forecasts()
    historical = load_historical()
    long_term_forecasts = load_long_term_forecasts()
    long_term_daily = load_long_term_daily()
    long_term_monthly = load_long_term_monthly()

    # Create tabs
    tab1, tab2 = st.tabs(["7-Day Forecast", "Long-Term Outlook"])

    with tab1:
        render_7day_forecast(forecasts, daily_forecasts, historical, long_term_forecasts)

    with tab2:
        render_long_term_outlook(long_term_daily, long_term_monthly, historical)


if __name__ == "__main__":
    main()
