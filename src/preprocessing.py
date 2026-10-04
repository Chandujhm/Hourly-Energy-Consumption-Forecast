from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PJMW_MW_Hourly (2).xlsx"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pjm_energy_cleaned.csv"
)


def load_raw_data() -> pd.DataFrame:
    """
    Load the original PJM hourly energy consumption dataset.
    """

    df = pd.read_excel(RAW_DATA_PATH)

    required_columns = {
        "Datetime",
        "PJMW_MW"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}"
        )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"]
    )

    return df


def clean_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply documented preprocessing steps.

    Steps:
    1. Sort observations chronologically.
    2. Preserve repeated timestamps because they occur
       around clock-change periods.
    3. Identify the isolated extreme anomaly.
    4. Replace the anomaly with the mean of the
       neighboring hourly observations.
    5. Record whether an observation was anomalous
       or imputed.
    """

    cleaned = df.copy()

    # Allow interpolated energy values to contain decimals.
    cleaned["PJMW_MW"] = cleaned["PJMW_MW"].astype(float)

    # Ensure chronological ordering.
    cleaned = (
        cleaned
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    # Track anomalous observations.
    cleaned["is_anomaly"] = False

    # Track values that were replaced/imputed.
    cleaned["is_imputed"] = False

    # Identified isolated extreme anomaly.
    anomaly_timestamp = pd.Timestamp(
        "2003-05-29 00:00:00"
    )

    cleaned.loc[
        cleaned["Datetime"] == anomaly_timestamp,
        "is_anomaly"
    ] = True

    # Retrieve the observations immediately before
    # and after the anomalous timestamp.
    previous_timestamp = (
        anomaly_timestamp
        - pd.Timedelta(hours=1)
    )

    next_timestamp = (
        anomaly_timestamp
        + pd.Timedelta(hours=1)
    )

    previous_values = cleaned.loc[
        cleaned["Datetime"] == previous_timestamp,
        "PJMW_MW"
    ]

    next_values = cleaned.loc[
        cleaned["Datetime"] == next_timestamp,
        "PJMW_MW"
    ]

    # Only perform interpolation when both neighboring
    # observations are available.
    if (
        not previous_values.empty
        and not next_values.empty
    ):

        previous_value = previous_values.iloc[0]
        next_value = next_values.iloc[0]

        interpolated_value = (
            previous_value + next_value
        ) / 2

        cleaned.loc[
            cleaned["Datetime"] == anomaly_timestamp,
            "PJMW_MW"
        ] = interpolated_value

        cleaned.loc[
            cleaned["Datetime"] == anomaly_timestamp,
            "is_imputed"
        ] = True

    return cleaned


def save_processed_data(
    df: pd.DataFrame
) -> None:
    """
    Save the processed dataset.
    """

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )


def main() -> None:
    """
    Run the preprocessing pipeline.
    """

    print(
        "Loading raw dataset..."
    )

    df = load_raw_data()

    print(
        f"Raw records: {len(df):,}"
    )

    cleaned_df = clean_data(df)

    print(
        f"Processed records: {len(cleaned_df):,}"
    )

    print(
        "Flagged anomalies:",
        int(cleaned_df["is_anomaly"].sum())
    )

    print(
        "Imputed values:",
        int(cleaned_df["is_imputed"].sum())
    )

    anomaly_rows = cleaned_df[
        cleaned_df["is_imputed"]
    ]

    if not anomaly_rows.empty:

        print(
            "\nImputed anomaly:"
        )

        print(
            anomaly_rows[
                [
                    "Datetime",
                    "PJMW_MW",
                    "is_anomaly",
                    "is_imputed"
                ]
            ].to_string(index=False)
        )

    save_processed_data(
        cleaned_df
    )

    print(
        f"\nSaved processed dataset to:\n"
        f"{PROCESSED_DATA_PATH}"
    )


if __name__ == "__main__":
    main()