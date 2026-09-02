import os

import mlflow

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)


def get_run_metrics(
    run_id: str,
) -> dict[str, float]:
    """
    Fetch evaluation metrics from an MLflow run.
    """

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

    run = mlflow.get_run(
        run_id
    )

    metrics = run.data.metrics

    required_metrics = {
        "test_mae",
        "test_rmse",
        "test_r2",
    }

    missing_metrics = (
        required_metrics
        - metrics.keys()
    )

    if missing_metrics:
        raise ValueError(
            
                "MLflow run is missing required "
                f"metrics: {sorted(missing_metrics)}"
            
        )

    return {
        "mae": float(
            metrics["test_mae"]
        ),
        "rmse": float(
            metrics["test_rmse"]
        ),
        "r2": float(
            metrics["test_r2"]
        ),
    }