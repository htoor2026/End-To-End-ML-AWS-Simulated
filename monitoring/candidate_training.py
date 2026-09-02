import os

import mlflow
import numpy as np
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from monitoring.retraining_dataset import (
    build_retraining_dataset,
)

MIN_RETRAINING_ROWS = 30
TEST_SIZE = 0.20
RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)


def configure_mlflow() -> None:
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )


def build_candidate_model(
    feature_columns: list[str],
) -> TransformedTargetRegressor:
    gr_liv_area_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "log_transform",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    remaining_columns = [
        column
        for column in feature_columns
        if column != "Gr Liv Area"
    ]

    remaining_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "gr_liv_area",
                gr_liv_area_pipeline,
                ["Gr Liv Area"],
            ),
            (
                "remaining_numeric",
                remaining_pipeline,
                remaining_columns,
            ),
        ],
        remainder="drop",
    )

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=regression_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_candidate() -> dict:
    configure_mlflow()

    dataset = build_retraining_dataset()

    if dataset.empty:
        return {
            "trained": False,
            "reason": (
                "No labelled production data available."
            ),
            "rows": 0,
        }

    if len(dataset) < MIN_RETRAINING_ROWS:
        return {
            "trained": False,
            "reason": (
                "Not enough labelled production data "
                "to train a candidate model."
            ),
            "rows": len(dataset),
            "minimum_required": (
                MIN_RETRAINING_ROWS
            ),
        }

    target_column = "SalePrice"

    X = dataset.drop(
        columns=[target_column],
    )

    y = dataset[
        target_column
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    model = build_candidate_model(
        feature_columns=X.columns.tolist()
    )

    model.fit(
        X_train,
        y_train,
    )

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    train_mae = mean_absolute_error(
        y_train,
        train_predictions,
    )

    train_rmse = mean_squared_error(
        y_train,
        train_predictions,
    ) ** 0.5

    train_r2 = r2_score(
        y_train,
        train_predictions,
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    test_rmse = mean_squared_error(
        y_test,
        test_predictions,
    ) ** 0.5

    test_r2 = r2_score(
        y_test,
        test_predictions,
    )

    with mlflow.start_run(
        run_name=(
            "production_retraining_candidate"
        )
    ) as run:
        mlflow.log_params(
            {
                "training_source": (
                    "production_labelled_data"
                ),
                "training_rows": len(dataset),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )

        mlflow.log_metrics(
            {
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        run_id = run.info.run_id

    return {
        "trained": True,
        "run_id": run_id,
        "rows": len(dataset),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": {
            "test_mae": float(
                test_mae
            ),
            "test_rmse": float(
                test_rmse
            ),
            "test_r2": float(
                test_r2
            ),
        },
    }


if __name__ == "__main__":
    result = train_candidate()

    print(
        "\nCANDIDATE TRAINING RESULT\n"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )