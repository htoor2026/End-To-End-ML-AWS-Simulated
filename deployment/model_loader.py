from functools import lru_cache
import os
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "champion_model.joblib"
)

MODEL_PATH = Path(
    os.getenv(
        "BUNDLED_MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
)


@lru_cache(maxsize=1)
def load_bundled_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Bundled model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_bundled(
    features: dict,
) -> float:
    model = load_bundled_model()

    dataframe = pd.DataFrame(
        [features]
    )

    prediction = model.predict(
        dataframe
    )

    return float(
        prediction[0]
    )
