import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PRODUCTION_SAMPLES = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "extracted_data"
    / "AmesHousing.csv"
)

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "predictions.jsonl"
)

MONITORED_FEATURES = [
    "Lot Area",
    "Overall Qual",
    "Year Built",
    "Gr Liv Area",
    "Bedroom AbvGr",
    "Garage Cars",
]


def load_reference_data() -> pd.DataFrame:
    """Load training/reference data."""
    return pd.read_csv(
        REFERENCE_DATA_PATH
    )


def load_production_data() -> pd.DataFrame:
    """Load production prediction logs."""

    records = []

    with PREDICTION_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            record = json.loads(line)

            production_record = (
                record["features"].copy()
            )

            production_record[
                "prediction"
            ] = record["prediction"]

            records.append(
                production_record
            )

    return pd.DataFrame(records)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Common interpretation:
        PSI < 0.10  -> little/no drift
        0.10-0.25   -> moderate drift
        PSI > 0.25  -> significant drift
    """

    reference = (
        pd.to_numeric(
            reference,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    current = (
        pd.to_numeric(
            current,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return np.nan

    breakpoints = np.quantile(
        reference,
        np.linspace(
            0,
            1,
            bins + 1,
        ),
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints,
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints,
    )

    reference_pct = (
        reference_counts
        / reference_counts.sum()
    )

    current_pct = (
        current_counts
        / current_counts.sum()
    )

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None,
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None,
    )

    psi = np.sum(
        (
            current_pct
            - reference_pct
        )
        * np.log(
            current_pct
            / reference_pct
        )
    )

    return float(psi)


def drift_status(
    psi: float,
) -> str:

    if np.isnan(psi):
        return "insufficient_data"

    if psi < 0.10:
        return "stable"

    if psi < 0.25:
        return "moderate_drift"

    return "significant_drift"

def overall_drift_status(report: pd.DataFrame) -> str:
    statuses = set(report["status"])

    if "significant_drift" in statuses:
        return "critical"

    if "moderate_drift" in statuses:
        return "warning"

    if statuses == {"insufficient_data"}:
        return "insufficient_data"

    return "healthy"


def generate_drift_report() -> pd.DataFrame:

    reference_df = load_reference_data()
    production_df = load_production_data()

    results = []

    production_sample_size = len(
        production_df
    )

    for feature in MONITORED_FEATURES:

        if (
            production_sample_size
            < MIN_PRODUCTION_SAMPLES
        ):
            results.append(
                {
            "feature": feature,
            "reference_mean": reference_df[
                feature
            ].mean(),
            "production_mean": production_df[
                feature
            ].mean(),
            "production_samples": production_sample_size,
            "psi": None,
            "status": "insufficient_data",
        }
    )
            continue

        psi = population_stability_index(
            reference_df[feature],
            production_df[feature],
        )

        results.append(
            {
                "feature": feature,
                "reference_mean": (
                    reference_df[
                        feature
                    ].mean()
                ),
                "production_mean": (
                    production_df[
                        feature
                    ].mean()
                ),
                "production_samples": (
                    production_sample_size
                ),
                "psi": psi,
                "status": drift_status(
                    psi
                ),
            }
        )

    return pd.DataFrame(results)
if __name__ == "__main__":

    report = generate_drift_report()

    print(
        "\nDATA DRIFT REPORT\n"
    )

    print(
        report.to_string(
            index=False
        )
    )