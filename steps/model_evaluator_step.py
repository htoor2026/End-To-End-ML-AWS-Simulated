import logging
from typing import Annotated, Tuple

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import ArtifactConfig, step
from zenml.client import Client

from src.model_evaluator import (
    ModelEvaluator,
    RegressionModelEvaluationStrategy,
)


# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def model_evaluator_step(
    trained_model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[
        dict,
        ArtifactConfig(name="evaluation_metrics"),
    ],
    Annotated[
        float,
        ArtifactConfig(name="mse"),
    ],
]:
    """
    Evaluates the trained regression model using test data.

    Parameters:
    trained_model (RegressorMixin): The trained regression model.
    X_test (pd.DataFrame): The testing feature dataset.
    y_test (pd.Series): The true testing target values.

    Returns:
    Tuple[dict, float]:
        A dictionary containing evaluation metrics
        and the Mean Squared Error value.
    """

    logging.info("Starting model evaluation.")

    # Initialize the regression evaluation strategy
    evaluator = ModelEvaluator(
        RegressionModelEvaluationStrategy()
    )

    # Evaluate the trained model
    evaluation_metrics = evaluator.evaluate(
        model=trained_model,
        X_test=X_test,
        y_test=y_test,
    )

    # Extract individual metrics
    mae = float(evaluation_metrics["MAE"])
    mse = float(evaluation_metrics["MSE"])
    rmse = float(evaluation_metrics["RMSE"])
    r2 = float(evaluation_metrics["R2"])

    logging.info("Logging evaluation metrics to MLflow.")

    # Log evaluation metrics to MLflow
    if mlflow.active_run():
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

    logging.info("Model evaluation completed successfully.")
    logging.info("Test MAE: %.4f", mae)
    logging.info("Test MSE: %.4f", mse)
    logging.info("Test RMSE: %.4f", rmse)
    logging.info("Test R2: %.4f", r2)

    return evaluation_metrics, mse