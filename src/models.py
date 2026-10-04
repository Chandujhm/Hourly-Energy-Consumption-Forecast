import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def naive_forecast(df: pd.DataFrame) -> pd.Series:
    """
    Predict the current value using the previous hour's consumption.
    """

    return df["PJMW_MW"].shift(1)


def create_random_forest() -> RandomForestRegressor:
    """
    Create the Random Forest forecasting model.
    """

    return RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def create_gradient_boosting() -> GradientBoostingRegressor:
    """
    Create the Gradient Boosting forecasting model.
    """

    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    from pathlib import Path

from evaluation import calculate_metrics
from features import create_features
from preprocessing import load_raw_data, clean_data


FEATURE_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "time_diff_hours",
    "is_regular_hour",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
]


def prepare_model_data():
    """
    Load, clean, and create model features.
    """

    df = load_raw_data()
    df = clean_data(df)
    df = create_features(df)

    # Remove rows where lag/rolling features are unavailable.
    model_df = df.dropna(
        subset=FEATURE_COLUMNS + ["PJMW_MW"]
    ).copy()

    return model_df


def split_last_year(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out the final year of observations as the test set.

    The split is chronological and is based on the final
    timestamp in the dataset.
    """

    last_timestamp = df["Datetime"].max()

    test_start = last_timestamp - pd.DateOffset(years=1)

    train_df = df[
        df["Datetime"] < test_start
    ].copy()

    test_df = df[
        df["Datetime"] >= test_start
    ].copy()

    return train_df, test_df


def train_models():
    """
    Train baseline and machine-learning forecasting models.
    """

    df = prepare_model_data()

    train_df, test_df = split_last_year(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["PJMW_MW"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["PJMW_MW"]

    results = {}

    # Baseline
    baseline_predictions = naive_forecast(test_df)

    baseline_mask = baseline_predictions.notna()

    results["Naive Previous Hour"] = calculate_metrics(
        y_test[baseline_mask],
        baseline_predictions[baseline_mask],
    )

    # Random Forest
    random_forest = create_random_forest()

    random_forest.fit(
        X_train,
        y_train,
    )
    model_path = Path(__file__).resolve().parents[1] / "models" / "random_forest_energy_forecast.joblib"

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        random_forest,
        model_path
    )

    print(
        f"\nSaved Random Forest model to:\n{model_path}"
    )

    rf_predictions = random_forest.predict(X_test)

    results["Random Forest"] = calculate_metrics(
        y_test,
        rf_predictions,
    )

    # Gradient Boosting
    gradient_boosting = create_gradient_boosting()

    gradient_boosting.fit(
        X_train,
        y_train,
    )

    gb_predictions = gradient_boosting.predict(X_test)

    results["Gradient Boosting"] = calculate_metrics(
        y_test,
        gb_predictions,
    )

    comparison = pd.DataFrame(results).T

    comparison.index.name = "Model"

    return (
        comparison,
        random_forest,
        gradient_boosting,
        train_df,
        test_df,
    )


if __name__ == "__main__":
    comparison, _, _, train_df, test_df = train_models()

    print("\nTraining completed.")
    print(f"Training records: {len(train_df):,}")
    print(f"Test records: {len(test_df):,}")

    print("\nModel Performance:")
    print(comparison.round(3))