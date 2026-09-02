import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
)

PREDICTIONS_LOG = (
    LOG_DIR
    / "predictions.jsonl"
)


def log_prediction(
    request_id: str,
    features: dict[str, Any],
    prediction: float,
) -> None:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now(
            UTC
        ).isoformat(),
        "request_id": request_id,
        "model_name": "prices_predictor",
        "features": features,
        "prediction": float(
            prediction
        ),
    }

    with PREDICTIONS_LOG.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(record)
            + "\n"
        )