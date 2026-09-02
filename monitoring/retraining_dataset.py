import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_LOG = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "predictions.jsonl"
)

ACTUALS_LOG = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "actuals.jsonl"
)


def load_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def build_retraining_dataset() -> pd.DataFrame:
    predictions = load_jsonl(
        PREDICTIONS_LOG
    )

    actuals = load_jsonl(
        ACTUALS_LOG
    )

    if not predictions:
        return pd.DataFrame()

    if not actuals:
        return pd.DataFrame()

    prediction_rows = []

    for record in predictions:
        row = {
            "request_id": record[
                "request_id"
            ],
        }

        features = record.get(
            "features",
            {},
        )

        row.update(features)

        prediction_rows.append(
            row
        )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    actual_df = pd.DataFrame(
        actuals
    )

    actual_df = actual_df[
        [
            "request_id",
            "actual_price",
        ]
    ]

    # Keep the latest actual if the same
    # request was submitted more than once.
    actual_df = (
        actual_df
        .drop_duplicates(
            subset=["request_id"],
            keep="last",
        )
    )

    retraining_df = prediction_df.merge(
        actual_df,
        on="request_id",
        how="inner",
    )

    retraining_df = (
        retraining_df.rename(
            columns={
                "actual_price": "SalePrice",
            }
        )
    )

    retraining_df = (
        retraining_df.drop(
            columns=["request_id"]
        )
    )

    return retraining_df


if __name__ == "__main__":
    dataset = (
        build_retraining_dataset()
    )

    print(
        "\nRETRAINING DATASET\n"
    )

    print(
        f"Rows: {len(dataset)}"
    )

    print(
        f"Columns: {len(dataset.columns)}"
    )

    if not dataset.empty:
        print(
            "\nColumns:"
        )

        print(
            dataset.columns.tolist()
        )

        print(
            "\nSample:"
        )

        print(
            dataset.head()
        )