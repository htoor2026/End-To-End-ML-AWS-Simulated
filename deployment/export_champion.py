import os
from pathlib import Path

import joblib
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "champion_model.joblib"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "prices_predictor",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)


def main() -> None:
    if not TRACKING_URI:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. "
            "Export the same tracking URI used by your local MLflow registry."
        )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print(
        f"Loading MLflow model: {model_uri}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        OUTPUT_PATH,
    )

    print(
        f"Bundled champion written to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
