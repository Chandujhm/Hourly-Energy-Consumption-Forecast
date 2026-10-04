import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series
) -> dict:
    """
    Calculate standard forecasting evaluation metrics.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    non_zero_mask = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[non_zero_mask] - predicted[non_zero_mask])
                / actual[non_zero_mask]
            )
        )
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """
    Convert model evaluation results into a comparison table.
    """

    comparison = pd.DataFrame(results).T

    comparison.index.name = "Model"

    return comparison.sort_values("RMSE")