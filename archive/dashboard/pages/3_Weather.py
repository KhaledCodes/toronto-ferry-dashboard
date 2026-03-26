"""
Weather Analysis Page
Explore how weather affects ferry ridership.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Weather Analysis", page_icon=":partly_sunny:", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent
HOURLY_DATA_PATH = PROJECT_ROOT / "outputs" / "hourly_data.csv"


@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(HOURLY_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


def main():
    st.title(":partly_sunny: Weather Analysis")
    st.markdown("Understand how weather conditions affect ferry ridership")

    df = load_data()

    # Filter to avoid outliers (remove zero ridership hours for better viz)
    df_nonzero = df[df['redemption_count'] > 0]

    # Temperature vs Ridership
    st.subheader("Temperature Impact")

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot
        sample_df = df_nonzero.sample(min(5000, len(df_nonzero)))
        fig_temp = px.scatter(
            sample_df,
            x='temp',
            y='redemption_count',
            color='is_weekend',
            opacity=0.5,
            title='Temperature vs Hourly Ridership',
            labels={
                'temp': 'Temperature (C)',
                'redemption_count': 'Redemptions',
                'is_weekend': 'Weekend'
            },
            color_discrete_map={0: '#1E88E5', 1: '#E91E63'}
        )
        fig_temp.update_layout(
            height=400,
            legend_title_text='Day Type',
            legend=dict(
                itemsizing='constant',
                title_font_size=12
            )
        )
        # Update legend labels
        fig_temp.for_each_trace(lambda t: t.update(name='Weekday' if t.name == '0' else 'Weekend'))
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        # Temperature bins
        df_nonzero['temp_bin'] = pd.cut(
            df_nonzero['temp'],
            bins=[-20, 0, 10, 20, 30, 40],
            labels=['Below 0', '0-10', '10-20', '20-30', '30+']
        )
        temp_avg = df_nonzero.groupby('temp_bin')['redemption_count'].mean().reset_index()

        fig_temp_bar = px.bar(
            temp_avg,
            x='temp_bin',
            y='redemption_count',
            title='Average Ridership by Temperature Range',
            labels={'temp_bin': 'Temperature Range (C)', 'redemption_count': 'Avg Redemptions'}
        )
        fig_temp_bar.update_traces(marker_color='#FF7043')
        fig_temp_bar.update_layout(height=400)
        st.plotly_chart(fig_temp_bar, use_container_width=True)

    st.markdown("---")

    # Precipitation Impact
    st.subheader("Precipitation Impact")

    col1, col2 = st.columns(2)

    with col1:
        # Rain vs no rain comparison
        df_nonzero['has_rain'] = df_nonzero['prcp'] > 0.1
        rain_comparison = df_nonzero.groupby('has_rain')['redemption_count'].mean().reset_index()
        rain_comparison['condition'] = rain_comparison['has_rain'].map({True: 'Rainy', False: 'Dry'})

        fig_rain = px.bar(
            rain_comparison,
            x='condition',
            y='redemption_count',
            title='Average Ridership: Rainy vs Dry',
            labels={'condition': 'Weather', 'redemption_count': 'Avg Redemptions'}
        )
        fig_rain.update_traces(marker_color=['#1E88E5', '#90CAF9'])
        fig_rain.update_layout(height=350)
        st.plotly_chart(fig_rain, use_container_width=True)

        # Calculate percentage difference
        dry_avg = rain_comparison[rain_comparison['condition'] == 'Dry']['redemption_count'].values[0]
        rain_avg = rain_comparison[rain_comparison['condition'] == 'Rainy']['redemption_count'].values[0]
        pct_diff = ((dry_avg - rain_avg) / dry_avg) * 100
        st.info(f"Ridership drops by **{pct_diff:.1f}%** on rainy days compared to dry days")

    with col2:
        # Precipitation amount vs ridership
        fig_prcp = px.scatter(
            df_nonzero[df_nonzero['prcp'] < 10].sample(min(3000, len(df_nonzero))),
            x='prcp',
            y='redemption_count',
            opacity=0.4,
            title='Precipitation Amount vs Ridership',
            labels={'prcp': 'Precipitation (mm)', 'redemption_count': 'Redemptions'}
        )
        fig_prcp.update_traces(marker_color='#42A5F5')
        fig_prcp.update_layout(height=350)
        st.plotly_chart(fig_prcp, use_container_width=True)

    st.markdown("---")

    # Correlation Matrix
    st.subheader("Weather Feature Correlations")

    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd', 'pres', 'redemption_count']
    corr_df = df_nonzero[weather_cols].dropna()
    corr_matrix = corr_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title='Correlation Matrix',
        labels=dict(color='Correlation'),
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Key insights
    col1, col2, col3 = st.columns(3)

    temp_corr = corr_matrix.loc['temp', 'redemption_count']
    rhum_corr = corr_matrix.loc['rhum', 'redemption_count']
    prcp_corr = corr_matrix.loc['prcp', 'redemption_count']

    with col1:
        st.metric("Temp Correlation", f"{temp_corr:.3f}", help="Higher temps = more ridership")

    with col2:
        st.metric("Humidity Correlation", f"{rhum_corr:.3f}", help="Higher humidity = less ridership")

    with col3:
        st.metric("Precip Correlation", f"{prcp_corr:.3f}", help="More rain = less ridership")

    st.markdown("---")

    # Wind Speed Impact
    st.subheader("Wind Speed Impact")

    df_nonzero['wind_bin'] = pd.cut(
        df_nonzero['wspd'],
        bins=[0, 10, 20, 30, 100],
        labels=['Calm (0-10)', 'Light (10-20)', 'Moderate (20-30)', 'Strong (30+)']
    )
    wind_avg = df_nonzero.groupby('wind_bin')['redemption_count'].mean().reset_index()

    fig_wind = px.bar(
        wind_avg,
        x='wind_bin',
        y='redemption_count',
        title='Average Ridership by Wind Speed',
        labels={'wind_bin': 'Wind Speed (km/h)', 'redemption_count': 'Avg Redemptions'}
    )
    fig_wind.update_traces(marker_color='#26A69A')
    st.plotly_chart(fig_wind, use_container_width=True)

    # Seasonal weather patterns
    st.markdown("---")
    st.subheader("Seasonal Weather Patterns")

    # Group by month and calculate averages
    monthly = df.groupby('month').agg({
        'redemption_count': 'mean',
        'temp': 'mean'
    }).reset_index()

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])

    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=monthly['month_name'],
        y=monthly['redemption_count'],
        name='Avg Redemptions',
        marker_color='#1E88E5',
        yaxis='y'
    ))
    fig_seasonal.add_trace(go.Scatter(
        x=monthly['month_name'],
        y=monthly['temp'],
        name='Avg Temp',
        mode='lines+markers',
        line=dict(color='#FF7043', width=3),
        yaxis='y2'
    ))

    fig_seasonal.update_layout(
        title='Monthly Ridership vs Temperature',
        yaxis=dict(title='Avg Hourly Redemptions', side='left'),
        yaxis2=dict(title='Avg Temperature (C)', side='right', overlaying='y'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)


if __name__ == "__main__":
    main()
