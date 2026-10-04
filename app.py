import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PJM Energy Consumption Forecast",
    page_icon="⚡",
    layout="wide",
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

HISTORICAL_DATA_PATH = (
    "data/processed/pjm_energy_cleaned.csv"
)

FORECAST_DATA_PATH = (
    "data/processed/30_day_energy_forecast.csv"
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_historical_data():
    df = pd.read_csv(
        HISTORICAL_DATA_PATH,
        parse_dates=["Datetime"],
    )

    return df.sort_values("Datetime").reset_index(
        drop=True
    )


@st.cache_data
def load_forecast_data():
    df = pd.read_csv(
        FORECAST_DATA_PATH,
        parse_dates=["Datetime"],
    )

    return df.sort_values("Datetime").reset_index(
        drop=True
    )


# --------------------------------------------------
# Load data
# --------------------------------------------------

historical_df = load_historical_data()
forecast_df = load_forecast_data()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("⚡ PJM Hourly Energy Consumption Forecast")

st.markdown(
    """
    **Professional Time-Series Forecasting Dashboard**

    Analyze historical PJM electricity consumption,
    evaluate forecasting models, and explore a
    30-day hourly energy-demand forecast.
    """
)

st.info(
    """
    **Forecast Model:** Random Forest Regressor  
    **Validation Strategy:** Chronological last-year holdout  
    **Forecast Horizon:** 30 days / 720 hours  
    **Key Features:** Calendar, holiday, lag, and rolling statistics
    """
)


# --------------------------------------------------
# KPI section
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Historical Records",
        f"{len(historical_df):,}",
    )

with col2:
    st.metric(
        "Average Consumption",
        f"{historical_df['PJMW_MW'].mean():,.0f} MW",
    )

with col3:
    st.metric(
        "Peak Consumption",
        f"{historical_df['PJMW_MW'].max():,.0f} MW",
    )

with col4:
    st.metric(
        "Forecast Horizon",
        f"{len(forecast_df):,} hours",
    )


# --------------------------------------------------
# Historical consumption
# --------------------------------------------------
st.header("Data Quality")

quality_col1, quality_col2, quality_col3, quality_col4 = (
    st.columns(4)
)

with quality_col1:
    st.metric(
        "Missing Values",
        f"{historical_df['PJMW_MW'].isna().sum():,}",
    )

with quality_col2:
    st.metric(
        "Duplicate Timestamps",
        f"{historical_df['Datetime'].duplicated().sum():,}",
    )

with quality_col3:
    st.metric(
        "Flagged Anomalies",
        f"{historical_df['is_anomaly'].sum():,}",
    )

with quality_col4:
    st.metric(
        "Imputed Values",
        f"{historical_df['is_imputed'].sum():,}",
    )

st.caption(
    "The raw source data is preserved separately. "
    "The processed dataset documents anomaly handling "
    "through explicit anomaly and imputation flags."
)

st.header("Historical Energy Consumption")

historical_chart = (
    historical_df
    .set_index("Datetime")["PJMW_MW"]
    .resample("D")
    .mean()
)

st.line_chart(
    historical_chart,
)


# --------------------------------------------------
# Daily consumption pattern
# --------------------------------------------------

st.header("Average Consumption by Hour")

hourly_profile = (
    historical_df
    .assign(
        Hour=historical_df["Datetime"].dt.hour
    )
    .groupby("Hour")["PJMW_MW"]
    .mean()
)

st.line_chart(
    hourly_profile,
)


# --------------------------------------------------
# Day-of-week pattern
# --------------------------------------------------

st.header("Average Consumption by Day of Week")

weekday_profile = (
    historical_df
    .assign(
        Day=historical_df["Datetime"].dt.day_name()
    )
    .groupby("Day")["PJMW_MW"]
    .mean()
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

weekday_profile = weekday_profile.reindex(
    weekday_order
)

st.bar_chart(
    weekday_profile,
)


# --------------------------------------------------
# 30-day forecast
# --------------------------------------------------
st.header("Model Performance")

MODEL_COMPARISON_PATH = (
    "reports/model_comparison.csv"
)

model_comparison = pd.read_csv(
    MODEL_COMPARISON_PATH
)

performance_col1, performance_col2, performance_col3 = (
    st.columns(3)
)

with performance_col1:
    st.metric(
        "Best RMSE",
        f"{model_comparison['RMSE'].min():.2f}",
    )

with performance_col2:
    st.metric(
        "Best MAE",
        f"{model_comparison['MAE'].min():.2f}",
    )

with performance_col3:
    st.metric(
        "Best MAPE",
        f"{model_comparison['MAPE'].min():.3f}%",
    )

st.dataframe(
    model_comparison,
    use_container_width=True,
)

st.bar_chart(
    model_comparison.set_index("Model")[["RMSE", "MAE"]]
)
st.header("Selected Forecast Model")

best_model = model_comparison.loc[
    model_comparison["RMSE"].idxmin()
]

model_col1, model_col2, model_col3, model_col4 = (
    st.columns(4)
)

with model_col1:
    st.metric(
        "Selected Model",
        best_model["Model"],
    )

with model_col2:
    st.metric(
        "Test MAE",
        f"{best_model['MAE']:.2f} MW",
    )

with model_col3:
    st.metric(
        "Test RMSE",
        f"{best_model['RMSE']:.2f} MW",
    )

with model_col4:
    st.metric(
        "Test MAPE",
        f"{best_model['MAPE']:.3f}%",
    )

st.caption(
    "The model with the lowest RMSE on the chronological "
    "last-year holdout is selected for forecasting."
)
st.header("30-Day Hourly Forecast")

forecast_chart = forecast_df[
    ["Datetime", "Forecast_MW"]
].set_index("Datetime")

st.line_chart(
    forecast_chart,
    y="Forecast_MW",
)


# --------------------------------------------------
# Forecast summary
# --------------------------------------------------

st.header("Forecast Summary")

forecast_col1, forecast_col2, forecast_col3 = (
    st.columns(3)
)

with forecast_col1:
    st.metric(
        "Average Forecast",
        f"{forecast_df['Forecast_MW'].mean():,.0f} MW",
    )

with forecast_col2:
    st.metric(
        "Forecast Peak",
        f"{forecast_df['Forecast_MW'].max():,.0f} MW",
    )

with forecast_col3:
    st.metric(
        "Forecast Minimum",
        f"{forecast_df['Forecast_MW'].min():,.0f} MW",
    )


# --------------------------------------------------
# Forecast table
# --------------------------------------------------

st.header("Forecast Data")

st.dataframe(
    forecast_df,
    use_container_width=True,
)


# --------------------------------------------------
# Download forecast
# --------------------------------------------------

csv_data = forecast_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download 30-Day Forecast CSV",
    data=csv_data,
    file_name="30_day_energy_forecast.csv",
    mime="text/csv",
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "PJM Hourly Energy Consumption Forecast | "
    "Data Science Project"
)