import pandas as pd
from holidays import US


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from the Datetime column.
    """

    data = df.copy()
    
    data["time_diff_hours"] = (
        data["Datetime"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    data["is_regular_hour"] = (
        data["time_diff_hours"] == 1
    ).astype(int)

    data["year"] = data["Datetime"].dt.year
    data["month"] = data["Datetime"].dt.month
    data["day"] = data["Datetime"].dt.day
    data["hour"] = data["Datetime"].dt.hour
    data["day_of_week"] = data["Datetime"].dt.dayofweek
    data["day_of_year"] = data["Datetime"].dt.dayofyear
    data["week_of_year"] = data["Datetime"].dt.isocalendar().week.astype(int)

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)
    
        # US federal holiday indicator
    us_holidays = US(
        years=range(
            data["year"].min(),
            data["year"].max() + 1
        )
    )

    data["is_holiday"] = (
        data["Datetime"].dt.date
        .isin(us_holidays)
        .astype(int)
    )

    return data


def create_lag_features(
    df: pd.DataFrame,
    target_column: str = "PJMW_MW"
) -> pd.DataFrame:
    """
    Create timestamp-aligned lag and rolling features.

    Lag values are matched using actual timestamps rather than
    row positions. This is important because the source data
    contains repeated timestamps and irregular time intervals.
    """

    data = df.copy()

    # Create a unique timestamp lookup for historical consumption.
    # Repeated timestamps are represented by their mean value.
    lag_lookup = (
        data.groupby("Datetime", as_index=False)[target_column]
        .mean()
        .rename(columns={target_column: "historical_value"})
    )

    # Previous hour
    hour_lookup = lag_lookup.rename(
        columns={"historical_value": "lag_1h"}
    )

    data["lag_1h"] = (
        data["Datetime"] - pd.Timedelta(hours=1)
    ).map(
        hour_lookup.set_index("Datetime")["lag_1h"]
    )

    # Previous day
    day_lookup = lag_lookup.rename(
        columns={"historical_value": "lag_24h"}
    )

    data["lag_24h"] = (
        data["Datetime"] - pd.Timedelta(hours=24)
    ).map(
        day_lookup.set_index("Datetime")["lag_24h"]
    )

    # Previous week
    week_lookup = lag_lookup.rename(
        columns={"historical_value": "lag_168h"}
    )

    data["lag_168h"] = (
        data["Datetime"] - pd.Timedelta(hours=168)
    ).map(
        week_lookup.set_index("Datetime")["lag_168h"]
    )

    # Time-based rolling features.
    # Only historical observations are included.
    historical_series = (
        data.set_index("Datetime")[target_column]
        .sort_index()
    )

    rolling_mean_24h = (
        historical_series
        .shift(1)
        .rolling("24h")
        .mean()
    )

    rolling_std_24h = (
        historical_series
        .shift(1)
        .rolling("24h")
        .std()
    )

    rolling_mean_168h = (
        historical_series
        .shift(1)
        .rolling("168h")
        .mean()
    )

    rolling_features = pd.DataFrame(
        {
            "Datetime": rolling_mean_24h.index,
            "rolling_mean_24h": rolling_mean_24h.values,
            "rolling_std_24h": rolling_std_24h.values,
            "rolling_mean_168h": rolling_mean_168h.values,
        }
    )

    data = data.merge(
        rolling_features,
        on="Datetime",
        how="left",
    )

    return data
    """
    Create historical lag and rolling features.

    All features use only past observations to avoid
    future-data leakage.
    """

    data = df.copy()

    # Previous hour
    data["lag_1h"] = data[target_column].shift(1)

    # Previous day
    data["lag_24h"] = data[target_column].shift(24)

    # Previous week
    data["lag_168h"] = data[target_column].shift(168)

    # Rolling statistics based only on previous observations
    data["rolling_mean_24h"] = (
        data[target_column]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    data["rolling_std_24h"] = (
        data[target_column]
        .shift(1)
        .rolling(window=24)
        .std()
    )

    data["rolling_mean_168h"] = (
        data[target_column]
        .shift(1)
        .rolling(window=168)
        .mean()
    )

    return data


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the complete feature set.
    """

    data = create_time_features(df)
    data = create_lag_features(data)

    return data