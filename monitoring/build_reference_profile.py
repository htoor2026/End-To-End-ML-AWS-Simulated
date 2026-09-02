import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "extracted_data"
    / "AmesHousing.csv"
)

REFERENCE_PROFILE_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "reference_profile.json"
)


MONITORED_FEATURES = [
    "Lot Area",
    "Overall Qual",
    "Year Built",
    "Gr Liv Area",
    "Bedroom AbvGr",
    "Garage Cars",
]


NUM_BINS = 10


def create_bin_edges(
    series: pd.Series,
) -> list[float]:
    clean_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    quantiles = np.linspace(
        0,
        1,
        NUM_BINS + 1,
    )

    edges = clean_series.quantile(
        quantiles
    ).to_numpy()

    edges = np.unique(
        edges
    )

    if len(edges) < 2:
        value = float(
            clean_series.iloc[0]
        )

        edges = np.array(
            [
                value - 1,
                value + 1,
            ]
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return [
        float(edge)
        for edge in edges
    ]


def calculate_distribution(
    series: pd.Series,
    bin_edges: list[float],
) -> list[float]:
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
        return [
            0.0
            for _ in counts
        ]

    proportions = (
        counts
        / total
    )

    return [
        float(value)
        for value in proportions
    ]


def build_reference_profile() -> dict:
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            
                "Reference dataset not found: "
                f"{REFERENCE_DATA_PATH}"
            
        )

    dataframe = pd.read_csv(
        REFERENCE_DATA_PATH
    )

    profile = {
        "source": "AmesHousing.csv",
        "features": {},
    }

    for feature in MONITORED_FEATURES:
        if feature not in dataframe.columns:
            raise ValueError(
                
                    "Reference dataset is missing "
                    f"feature: {feature}"
                
            )

        series = dataframe[
            feature
        ]

        clean_series = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        bin_edges = create_bin_edges(
            clean_series
        )

        expected_distribution = (
            calculate_distribution(
                clean_series,
                bin_edges,
            )
        )

        profile["features"][
            feature
        ] = {
            "mean": float(
                clean_series.mean()
            ),
            "samples": len(clean_series),
            "bin_edges": (
                bin_edges
            ),
            "expected_distribution": (
                expected_distribution
            ),
        }

    return profile


def save_reference_profile() -> None:
    profile = (
        build_reference_profile()
    )

    with REFERENCE_PROFILE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            profile,
            file,
            indent=2,
        )

    print(
        
            "Reference profile written to: "
            f"{REFERENCE_PROFILE_PATH}"
        
    )


if __name__ == "__main__":
    save_reference_profile()