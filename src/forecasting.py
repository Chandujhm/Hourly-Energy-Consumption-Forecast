from pathlib import Path

import joblib
import pandas as pd
from holidays import US


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_energy_forecast.joblib"
)


def load_model():
    """Load the trained Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def generate_future_timestamps(
    last_timestamp: pd.Timestamp,
    periods: int = 24 * 30
) -> pd.DatetimeIndex:
    """Generate future hourly timestamps."""

    return pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=periods,
        freq="h",
    )


def create_forecast_dataframe(
    timestamps: pd.DatetimeIndex,
    predictions
) -> pd.DataFrame:
    """Create the final forecast dataframe."""

    return pd.DataFrame(
        {
            "Datetime": timestamps,
            "Forecast_MW": predictions,
        }
    )


def save_forecast(
    forecast: pd.DataFrame
) -> Path:
    """Save the 30-day forecast."""

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "30_day_energy_forecast.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast.to_csv(
        output_path,
        index=False
    )

    return output_path


def recursive_forecast(
    model,
    history: pd.DataFrame,
    periods: int = 24 * 30
) -> pd.DataFrame:
    """
    Generate a recursive multi-step hourly forecast.

    Each predicted hour is added to the historical series so
    subsequent predictions can use the newly predicted value.
    """

    history = history.copy()

    history["Datetime"] = pd.to_datetime(
        history["Datetime"]
    )

    history = history.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    target_column = "PJMW_MW"
    us_holidays = US(
        years=range(
            history["Datetime"].dt.year.min(),
            history["Datetime"].dt.year.max() + 2
        )
    )
    
    future_timestamps = generate_future_timestamps(
        history["Datetime"].max(),
        periods,
    )

    predictions = []

    for timestamp in future_timestamps:

        previous_hour = timestamp - pd.Timedelta(hours=1)
        previous_day = timestamp - pd.Timedelta(hours=24)
        previous_week = timestamp - pd.Timedelta(hours=168)

        lag_1h = (
            history.loc[
                history["Datetime"] == previous_hour,
                target_column
            ].mean()
        )

        lag_24h = (
            history.loc[
                history["Datetime"] == previous_day,
                target_column
            ].mean()
        )

        lag_168h = (
            history.loc[
                history["Datetime"] == previous_week,
                target_column
            ].mean()
        )

        recent_24h = history[
            history["Datetime"] < timestamp
        ].tail(24)[target_column]

        recent_168h = history[
            history["Datetime"] < timestamp
        ].tail(168)[target_column]

        if (
            pd.isna(lag_1h)
            or pd.isna(lag_24h)
            or pd.isna(lag_168h)
            or len(recent_24h) < 24
            or len(recent_168h) < 168
        ):
            raise ValueError(
                f"Insufficient historical data for {timestamp}"
            )

        feature_row = pd.DataFrame(
            {
                "year": [timestamp.year],
                "month": [timestamp.month],
                "day": [timestamp.day],
                "hour": [timestamp.hour],
                "day_of_week": [timestamp.dayofweek],
                "day_of_year": [timestamp.dayofyear],
                "week_of_year": [
                    int(timestamp.isocalendar().week)
                ],
                "is_weekend": [
                    int(timestamp.dayofweek >= 5)
                ],
                "is_holiday": [
                    int(timestamp.date() in us_holidays)
                ],
                "time_diff_hours": [1.0],
                "is_regular_hour": [1],
                "lag_1h": [lag_1h],
                "lag_24h": [lag_24h],
                "lag_168h": [lag_168h],
                "rolling_mean_24h": [
                    recent_24h.mean()
                ],
                "rolling_std_24h": [
                    recent_24h.std()
                ],
                "rolling_mean_168h": [
                    recent_168h.mean()
                ],
            }
        )

        prediction = model.predict(
            feature_row
        )[0]

        predictions.append(prediction)

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "Datetime": [timestamp],
                        target_column: [prediction],
                    }
                ),
            ],
            ignore_index=True,
        )

    return create_forecast_dataframe(
        future_timestamps,
        predictions,
    )


if __name__ == "__main__":

    processed_data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pjm_energy_cleaned.csv"
    )

    print("Loading processed energy data...")

    history = pd.read_csv(
        processed_data_path,
        parse_dates=["Datetime"]
    )

    print(
        f"Historical records: {len(history):,}"
    )

    print("Loading trained Random Forest model...")

    model = load_model()

    print("Generating 30-day forecast...")

    forecast = recursive_forecast(
        model=model,
        history=history,
        periods=24 * 30,
    )

    output_path = save_forecast(
        forecast
    )

    print(
        f"\nForecast records: {len(forecast):,}"
    )

    print(
        f"Forecast start: {forecast['Datetime'].min()}"
    )

    print(
        f"Forecast end:   {forecast['Datetime'].max()}"
    )

    print(
        f"\nSaved forecast to:\n{output_path}"
    )

    print("\nFirst 10 predictions:")

    print(
        forecast.head(10).to_string(index=False)
    )