import os

import mlflow
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from monitoring.retraining_dataset import (
    build_retraining_dataset,
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)

MIN_EVALUATION_ROWS = 10


def configure_mlflow() -> None:
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )


def evaluate_run_on_production_data(
    run_id: str,
) -> dict:
    configure_mlflow()

    dataset = build_retraining_dataset()

    if dataset.empty:
        return {
            "evaluated": False,
            "reason": (
                "No labelled production data available."
            ),
            "rows": 0,
        }

    if len(dataset) < MIN_EVALUATION_ROWS:
        return {
            "evaluated": False,
            "reason": (
                "Not enough labelled production data "
                "for shared model evaluation."
            ),
            "rows": len(dataset),
            "minimum_required": (
                MIN_EVALUATION_ROWS
            ),
        }

    target_column = "SalePrice"

    X = dataset.drop(
        columns=[target_column],
    )

    y = dataset[
        target_column
    ]

    model_uri = (
        f"runs:/{run_id}/model"
    )

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    predictions = model.predict(
        X
    )

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = mean_squared_error(
        y,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions,
    )

    return {
        "evaluated": True,
        "run_id": run_id,
        "rows": len(dataset),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
    }