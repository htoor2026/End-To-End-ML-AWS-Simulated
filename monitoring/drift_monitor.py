import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PROFILE_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "reference_profile.json"
)

PREDICTIONS_LOG_PATH = (
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


MIN_PRODUCTION_SAMPLES = 30

EPSILON = 1e-6


def load_reference_profile() -> dict:
    if not REFERENCE_PROFILE_PATH.exists():
        raise FileNotFoundError(
            
                "Reference profile not found: "
                f"{REFERENCE_PROFILE_PATH}"
            
        )

    with REFERENCE_PROFILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def load_production_data() -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        return pd.DataFrame()

    records = []

    with PREDICTIONS_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            features = record.get(
                "features",
                {},
            )

            records.append(
                features
            )

    return pd.DataFrame(
        records
    )


def calculate_production_distribution(
    series: pd.Series,
    bin_edges: list[float],
) -> np.ndarray:
    clean_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    counts, _ = np.histogram(
        clean_series,
        bins=bin_edges,
    )

    total = counts.sum()

    if total == 0:
        return np.zeros(
            len(counts),
            dtype=float,
        )

    return (
        counts
        / total
    )


def calculate_psi(
    expected_distribution: list[float],
    actual_distribution: np.ndarray,
) -> float:
    expected = np.asarray(
        expected_distribution,
        dtype=float,
    )

    actual = np.asarray(
        actual_distribution,
        dtype=float,
    )

    if len(expected) != len(actual):
        raise ValueError(
            
                "Expected and actual distributions "
                "must contain the same number of bins."
            
        )

    expected = np.clip(
        expected,
        EPSILON,
        None,
    )

    actual = np.clip(
        actual,
        EPSILON,
        None,
    )

    psi_values = (
        actual
        - expected
    ) * np.log(
        actual
        / expected
    )

    return float(
        np.sum(
            psi_values
        )
    )


def classify_psi(
    psi: float,
) -> str:
    if psi < 0.1:
        return "no_significant_drift"

    if psi < 0.25:
        return "moderate_drift"

    return "significant_drift"


def generate_drift_report() -> pd.DataFrame:
    reference_profile = (
        load_reference_profile()
    )

    production_data = (
        load_production_data()
    )

    report_rows = []

    for feature in MONITORED_FEATURES:
        reference_feature = (
            reference_profile[
                "features"
            ][feature]
        )

        reference_mean = (
            reference_feature[
                "mean"
            ]
        )

        if (
            production_data.empty
            or feature
            not in production_data.columns
        ):
            report_rows.append(
                {
                    "feature": feature,
                    "reference_mean": (
                        reference_mean
                    ),
                    "production_mean": None,
                    "production_samples": 0,
                    "psi": None,
                    "status": (
                        "insufficient_data"
                    ),
                }
            )

            continue

        production_series = (
            pd.to_numeric(
                production_data[
                    feature
                ],
                errors="coerce",
            ).dropna()
        )

        production_samples = (
            len(
                production_series
            )
        )

        production_mean = (
            float(
                production_series.mean()
            )
            if production_samples > 0
            else None
        )

        if (
            production_samples
            < MIN_PRODUCTION_SAMPLES
        ):
            report_rows.append(
                {
                    "feature": feature,
                    "reference_mean": (
                        reference_mean
                    ),
                    "production_mean": (
                        production_mean
                    ),
                    "production_samples": (
                        production_samples
                    ),
                    "psi": None,
                    "status": (
                        "insufficient_data"
                    ),
                }
            )

            continue

        bin_edges = [
            float(value)
            for value in (
                reference_feature[
                    "bin_edges"
                ]
            )
        ]

        expected_distribution = (
            reference_feature[
                "expected_distribution"
            ]
        )

        actual_distribution = (
            calculate_production_distribution(
                production_series,
                bin_edges,
            )
        )

        psi = calculate_psi(
            expected_distribution=(
                expected_distribution
            ),
            actual_distribution=(
                actual_distribution
            ),
        )

        report_rows.append(
            {
                "feature": feature,
                "reference_mean": (
                    reference_mean
                ),
                "production_mean": (
                    production_mean
                ),
                "production_samples": (
                    production_samples
                ),
                "psi": psi,
                "status": (
                    classify_psi(
                        psi
                    )
                ),
            }
        )

    return pd.DataFrame(
        report_rows
    )


def overall_drift_status(
    report: pd.DataFrame,
) -> str:
    if report.empty:
        return "insufficient_data"

    statuses = set(
        report[
            "status"
        ].tolist()
    )

    if (
        "significant_drift"
        in statuses
    ):
        return "critical"

    if (
        "moderate_drift"
        in statuses
    ):
        return "warning"

    if statuses == {
        "insufficient_data"
    }:
        return "insufficient_data"

    return "healthy"


if __name__ == "__main__":
    drift_report = (
        generate_drift_report()
    )

    print(
        "\nDRIFT REPORT\n"
    )

    print(
        drift_report.to_string(
            index=False
        )
    )

    print(
        "\nOVERALL STATUS:"
    )

    print(
        overall_drift_status(
            drift_report
        )
    )