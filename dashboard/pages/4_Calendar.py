"""
Calendar Patterns Page
Explore ridership patterns by time of day, day of week, and season.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Calendar Patterns", page_icon=":calendar:", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent
HOURLY_DATA_PATH = PROJECT_ROOT / "outputs" / "hourly_data.csv"


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(HOURLY_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


def main():
    st.title(":calendar: Calendar Patterns")
    st.markdown("Discover when ridership peaks and valleys occur")

    df = load_data()

    # Filters
    st.sidebar.header("Filters")

    years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Years",
        years,
        default=years[-3:]  # Last 3 years by default
    )

    if selected_years:
        df = df[df['year'].isin(selected_years)]

    # Hour x Day of Week Heatmap
    st.subheader("Hour x Day of Week Heatmap")

    pivot = df.pivot_table(
        values='redemption_count',
        index='hour_of_day',
        columns='day_of_week',
        aggfunc='mean'
    )

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Day of Week", y="Hour of Day", color="Avg Redemptions"),
        x=day_names,
        y=list(range(24)),
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.info("""
    **Key Insights:**
    - Peak hours are typically mid-day (10 AM - 4 PM)
    - Weekends (Sat/Sun) show significantly higher ridership
    - Summer weekends at noon are the busiest times
    """)

    st.markdown("---")

    # Hourly patterns by day type
    st.subheader("Hourly Patterns: Weekday vs Weekend")

    col1, col2 = st.columns(2)

    with col1:
        hourly_weekday = df[df['is_weekend'] == 0].groupby('hour_of_day')['redemption_count'].mean()
        hourly_weekend = df[df['is_weekend'] == 1].groupby('hour_of_day')['redemption_count'].mean()

        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hourly_weekday.index,
            y=hourly_weekday.values,
            mode='lines+markers',
            name='Weekday',
            line=dict(color='#1E88E5', width=3)
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly_weekend.index,
            y=hourly_weekend.values,
            mode='lines+markers',
            name='Weekend',
            line=dict(color='#E91E63', width=3)
        ))
        fig_hourly.update_layout(
            title='Average Hourly Pattern',
            xaxis_title='Hour of Day',
            yaxis_title='Avg Redemptions',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col2:
        # Day of week bar chart
        dow_avg = df.groupby('day_of_week')['redemption_count'].mean().reset_index()
        dow_avg['day_name'] = dow_avg['day_of_week'].map(dict(enumerate(day_names)))

        fig_dow = px.bar(
            dow_avg,
            x='day_name',
            y='redemption_count',
            title='Average Hourly Ridership by Day',
            labels={'day_name': 'Day', 'redemption_count': 'Avg Redemptions'}
        )
        fig_dow.update_traces(marker_color=['#1E88E5']*5 + ['#E91E63']*2)
        fig_dow.update_layout(height=400)
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("---")

    # Holiday Analysis
    st.subheader("Holiday Impact")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Holiday vs non-holiday
        holiday_avg = df.groupby('is_holiday')['redemption_count'].mean().reset_index()
        holiday_avg['type'] = holiday_avg['is_holiday'].map({0: 'Regular Day', 1: 'Holiday'})

        fig_holiday = px.bar(
            holiday_avg,
            x='type',
            y='redemption_count',
            title='Holiday vs Regular Day',
            labels={'type': '', 'redemption_count': 'Avg Redemptions'}
        )
        fig_holiday.update_traces(marker_color=['#1E88E5', '#FF7043'])
        fig_holiday.update_layout(height=350)
        st.plotly_chart(fig_holiday, use_container_width=True)

        # Calculate boost percentage
        regular = holiday_avg[holiday_avg['type'] == 'Regular Day']['redemption_count'].values[0]
        holiday = holiday_avg[holiday_avg['type'] == 'Holiday']['redemption_count'].values[0]
        boost = ((holiday - regular) / regular) * 100
        st.success(f"Holidays show **{boost:.1f}%** higher ridership!")

    with col2:
        # School break impact
        break_avg = df.groupby('is_school_break')['redemption_count'].mean().reset_index()
        break_avg['type'] = break_avg['is_school_break'].map({0: 'School Session', 1: 'School Break'})

        covid_avg = df.groupby('is_covid_lockdown')['redemption_count'].mean().reset_index()
        covid_avg['type'] = covid_avg['is_covid_lockdown'].map({0: 'Normal', 1: 'COVID Lockdown'})

        fig_special = go.Figure()

        # School breaks - check if both values exist
        school_session_val = break_avg[break_avg['is_school_break']==0]['redemption_count'].values
        school_break_val = break_avg[break_avg['is_school_break']==1]['redemption_count'].values
        if len(school_session_val) > 0 and len(school_break_val) > 0:
            fig_special.add_trace(go.Bar(
                x=['School Session', 'School Break'],
                y=[school_session_val[0], school_break_val[0]],
                name='School Breaks',
                marker_color='#43A047'
            ))

        # COVID impact - check if both values exist
        normal_val = covid_avg[covid_avg['is_covid_lockdown']==0]['redemption_count'].values
        lockdown_val = covid_avg[covid_avg['is_covid_lockdown']==1]['redemption_count'].values
        if len(normal_val) > 0 and len(lockdown_val) > 0:
            fig_special.add_trace(go.Bar(
                x=['Normal', 'COVID Lockdown'],
                y=[normal_val[0], lockdown_val[0]],
                name='COVID Impact',
                marker_color='#F44336'
            ))
        elif len(normal_val) > 0:
            # Only normal period data exists (no lockdown in selected range)
            fig_special.add_trace(go.Bar(
                x=['Normal Period'],
                y=[normal_val[0]],
                name='COVID Impact',
                marker_color='#F44336'
            ))

        fig_special.update_layout(
            title='Special Period Impact',
            yaxis_title='Avg Redemptions',
            barmode='group',
            height=350
        )
        st.plotly_chart(fig_special, use_container_width=True)

    st.markdown("---")

    # Monthly patterns
    st.subheader("Monthly Patterns")

    monthly = df.groupby('month')['redemption_count'].mean().reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])

    # Color by season
    season_colors = {
        'Jan': '#90CAF9', 'Feb': '#90CAF9', 'Dec': '#90CAF9',  # Winter - light blue
        'Mar': '#A5D6A7', 'Apr': '#A5D6A7', 'May': '#A5D6A7',  # Spring - light green
        'Jun': '#FFCC80', 'Jul': '#FFCC80', 'Aug': '#FFCC80',  # Summer - orange
        'Sep': '#FFAB91', 'Oct': '#FFAB91', 'Nov': '#FFAB91'   # Fall - light red
    }

    fig_monthly = px.bar(
        monthly,
        x='month_name',
        y='redemption_count',
        title='Average Hourly Ridership by Month',
        labels={'month_name': 'Month', 'redemption_count': 'Avg Redemptions'},
        color='month_name',
        color_discrete_map=season_colors
    )
    fig_monthly.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Annual trend
    st.markdown("---")
    st.subheader("Year-over-Year Daily Totals")

    daily = df.groupby(['year', 'date'])['redemption_count'].sum().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])

    fig_annual = px.box(
        daily,
        x='year',
        y='redemption_count',
        title='Distribution of Daily Ridership by Year',
        labels={'year': 'Year', 'redemption_count': 'Daily Redemptions'}
    )
    fig_annual.update_traces(marker_color='#7E57C2')
    st.plotly_chart(fig_annual, use_container_width=True)

    # Summary statistics by year
    st.subheader("Summary by Year")

    yearly_stats = daily.groupby('year').agg({
        'redemption_count': ['sum', 'mean', 'max']
    }).round(0)
    yearly_stats.columns = ['Total', 'Daily Avg', 'Peak Day']
    yearly_stats = yearly_stats.astype(int)

    st.dataframe(
        yearly_stats.style.format('{:,}'),
        use_container_width=True
    )


if __name__ == "__main__":
    main()
