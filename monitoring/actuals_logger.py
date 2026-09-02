import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "predictions.jsonl"
)

ACTUALS_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "actuals.jsonl"
)


def find_prediction(
    request_id: str,
) -> dict | None:
    if not PREDICTION_LOG_PATH.exists():
        return None

    with PREDICTION_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(line)

            if record["request_id"] == request_id:
                return record

    return None


def log_actual(
    request_id: str,
    actual_price: float,
) -> dict:
    prediction_record = find_prediction(
        request_id
    )

    if prediction_record is None:
        raise ValueError(
            "Prediction request_id not found."
        )

    ACTUALS_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "request_id": request_id,
        "predicted_price": (
            prediction_record[
                "prediction"
            ]
        ),
        "actual_price": actual_price,
    }

    with ACTUALS_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(record)
            + "\n"
        )

    return record