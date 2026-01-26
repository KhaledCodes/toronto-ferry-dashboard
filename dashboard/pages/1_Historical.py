"""
Historical Trends Page
View historical ridership data with various aggregations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Historical Trends", page_icon=":chart_with_upwards_trend:", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent
HOURLY_DATA_PATH = PROJECT_ROOT / "outputs" / "hourly_data.csv"


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(HOURLY_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


def main():
    st.title(":chart_with_upwards_trend: Historical Trends")
    st.markdown("Explore ferry ridership patterns over time")

    df = load_data()

    # Filters in sidebar
    st.sidebar.header("Filters")

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        df_filtered = df

    # Aggregation level
    agg_level = st.sidebar.selectbox(
        "Aggregation",
        ["Monthly", "Yearly", "Weekly", "Daily"],
        index=0
    )

    # Metric selection
    metric = st.sidebar.selectbox(
        "Metric",
        ["Redemptions", "Sales"],
        index=0
    )
    metric_col = 'redemption_count' if metric == "Redemptions" else 'sales_count'

    # Aggregate data based on selection
    if agg_level == "Monthly":
        df_filtered['month_start'] = df_filtered['timestamp'].dt.to_period('M').dt.start_time
        agg_df = df_filtered.groupby('month_start').agg({
            'redemption_count': 'sum',
            'sales_count': 'sum',
            'temp': 'mean'
        }).reset_index()
        x_col = 'month_start'
        title_suffix = "Monthly"
    elif agg_level == "Yearly":
        df_filtered['year'] = df_filtered['timestamp'].dt.year
        agg_df = df_filtered.groupby('year').agg({
            'redemption_count': 'sum',
            'sales_count': 'sum',
            'temp': 'mean'
        }).reset_index()
        x_col = 'year'
        title_suffix = "Yearly"
    elif agg_level == "Weekly":
        df_filtered['week'] = df_filtered['timestamp'].dt.to_period('W').dt.start_time
        agg_df = df_filtered.groupby('week').agg({
            'redemption_count': 'sum',
            'sales_count': 'sum',
            'temp': 'mean'
        }).reset_index()
        x_col = 'week'
        title_suffix = "Weekly"
    else:  # Daily
        agg_df = df_filtered.groupby('date').agg({
            'redemption_count': 'sum',
            'sales_count': 'sum',
            'temp': 'mean'
        }).reset_index()
        x_col = 'date'
        title_suffix = "Daily"

    # Main time series chart
    st.subheader(f"{title_suffix} {metric}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg_df[x_col],
        y=agg_df[metric_col],
        name=metric,
        marker_color='#1E88E5'
    ))

    # Add trendline (polynomial degree 3 for smooth curve)
    # Exclude incomplete periods (current year/month) from trendline calculation
    trendline_df = agg_df.copy()
    if agg_level == "Yearly":
        # Exclude current year (incomplete)
        current_year = datetime.now().year
        trendline_df = trendline_df[trendline_df[x_col] < current_year]
    elif agg_level == "Monthly":
        # Exclude current month (incomplete)
        current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        trendline_df = trendline_df[trendline_df[x_col] < current_month_start]

    if len(trendline_df) > 3:
        x_numeric = np.arange(len(trendline_df))
        z = np.polyfit(x_numeric, trendline_df[metric_col], 3)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=trendline_df[x_col],
            y=p(x_numeric),
            mode='lines',
            name='Trend',
            line=dict(color='#FF5722', width=1.5, dash='dash')
        ))

    # Add COVID and Flooding period indicators
    # COVID: March 2020 - December 2021
    # Flooding: 2017 season
    max_y = agg_df[metric_col].max()

    # COVID period shading
    fig.add_vrect(
        x0="2020-03-01" if agg_level != "Yearly" else 2020,
        x1="2021-12-31" if agg_level != "Yearly" else 2021,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="COVID",
        annotation_position="top left",
        annotation=dict(font_size=10, font_color="red")
    )

    # Flooding period shading (2017)
    fig.add_vrect(
        x0="2017-01-01" if agg_level != "Yearly" else 2017,
        x1="2017-12-31" if agg_level != "Yearly" else 2017,
        fillcolor="rgba(0, 100, 255, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Flooding",
        annotation_position="top left",
        annotation=dict(font_size=10, font_color="blue")
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric,
        hovermode='x unified',
        height=400,
        xaxis=dict(
            dtick="M12",
            tickformat="%Y"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Year-over-Year comparison
    st.markdown("---")
    st.subheader("Year-over-Year Comparison")

    # Get unique years
    df_filtered['year'] = df_filtered['timestamp'].dt.year
    df_filtered['day_of_year'] = df_filtered['timestamp'].dt.dayofyear

    years = sorted(df_filtered['year'].unique())[-4:]  # Last 4 years

    if len(years) > 1:
        yoy_data = df_filtered[df_filtered['year'].isin(years)].groupby(
            ['year', 'day_of_year']
        ).agg({metric_col: 'sum'}).reset_index()

        fig_yoy = px.line(
            yoy_data,
            x='day_of_year',
            y=metric_col,
            color='year',
            title=f'{metric} by Day of Year',
            labels={'day_of_year': 'Day of Year', metric_col: metric, 'year': 'Year'}
        )
        fig_yoy.update_layout(height=400)
        st.plotly_chart(fig_yoy, use_container_width=True)

    # Summary statistics
    st.markdown("---")
    st.subheader("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = int(agg_df[metric_col].sum())
        st.metric("Total", f"{total:,}")

    with col2:
        avg = int(agg_df[metric_col].mean())
        st.metric("Average", f"{avg:,}")

    with col3:
        peak = int(agg_df[metric_col].max())
        st.metric("Peak", f"{peak:,}")

    with col4:
        min_val = int(agg_df[metric_col].min())
        st.metric("Minimum", f"{min_val:,}")

    # Monthly breakdown
    st.markdown("---")
    st.subheader("Monthly Breakdown")

    df_filtered['month_name'] = df_filtered['timestamp'].dt.month_name()
    monthly_avg = df_filtered.groupby('month_name')[metric_col].mean().reset_index()

    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_avg['month_order'] = monthly_avg['month_name'].apply(lambda x: month_order.index(x))
    monthly_avg = monthly_avg.sort_values('month_order')

    fig_monthly = px.bar(
        monthly_avg,
        x='month_name',
        y=metric_col,
        title=f'Average Hourly {metric} by Month',
        labels={'month_name': 'Month', metric_col: f'Avg {metric}'}
    )
    fig_monthly.update_traces(marker_color='#43A047')
    st.plotly_chart(fig_monthly, use_container_width=True)


if __name__ == "__main__":
    main()
