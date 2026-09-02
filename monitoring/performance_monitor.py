import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

PERFORMANCE_WINDOW_SIZE = 50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTUALS_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "actuals.jsonl"
)

BASELINE_MAE = 13044.81
BASELINE_RMSE = 18626.84
BASELINE_R2 = 0.9221

MIN_PERFORMANCE_SAMPLES = 10


def load_actuals() -> pd.DataFrame:
    if not ACTUALS_LOG_PATH.exists():
        return pd.DataFrame()

    records = []

    with ACTUALS_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            records.append(
                json.loads(line)
            )

    dataframe = pd.DataFrame(
    records
)

    if dataframe.empty:
        return dataframe

    return dataframe.tail(
        PERFORMANCE_WINDOW_SIZE
).reset_index(
    drop=True
)


def calculate_performance() -> dict:
    df = load_actuals()

    if len(df) < MIN_PERFORMANCE_SAMPLES:
        return {
            "status": "insufficient_data",
            "samples": len(df),
            "minimum_samples_required": MIN_PERFORMANCE_SAMPLES,
            "mae": None,
            "rmse": None,
            "r2": None,
            "within_10k_pct": None,
            "within_20k_pct": None,
            "within_40k_pct": None,
            "avg_overprediction": None,
            "avg_underprediction": None,
            "baseline_mae": BASELINE_MAE,
            "baseline_rmse": BASELINE_RMSE,
            "baseline_r2": BASELINE_R2,
        }

    y_true = pd.to_numeric(
        df["actual_price"],
        errors="coerce",
    )

    y_pred = pd.to_numeric(
        df["predicted_price"],
        errors="coerce",
    )

    valid_mask = (
        y_true.notna()
        & y_pred.notna()
    )

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) < MIN_PERFORMANCE_SAMPLES:
        return {
            "status": "insufficient_data",
            "samples": len(y_true),
            "minimum_samples_required": MIN_PERFORMANCE_SAMPLES,
            "mae": None,
            "rmse": None,
            "r2": None,
            "within_10k_pct": None,
            "within_20k_pct": None,
            "within_40k_pct": None,
            "avg_overprediction": None,
            "avg_underprediction": None,
            "baseline_mae": BASELINE_MAE,
            "baseline_rmse": BASELINE_RMSE,
            "baseline_r2": BASELINE_R2,
        }

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    absolute_errors = np.abs(
        y_true - y_pred
    )

    within_10k = (
        absolute_errors <= 10000
    ).mean() * 100

    within_20k = (
        absolute_errors <= 20000
    ).mean() * 100

    within_40k = (
        absolute_errors <= 40000
    ).mean() * 100

    signed_errors = (
        y_pred - y_true
    )

    overprediction_errors = (
        signed_errors[
            signed_errors > 0
        ]
    )

    underprediction_errors = (
        signed_errors[
            signed_errors < 0
        ]
    )

    avg_overprediction = (
        overprediction_errors.mean()
        if len(overprediction_errors) > 0
        else 0.0
    )

    avg_underprediction = (
        abs(
            underprediction_errors.mean()
        )
        if len(underprediction_errors) > 0
        else 0.0
    )

    status = "healthy"

    if mae > BASELINE_MAE * 1.20:
        status = "degraded"

    return {
        "status": status,
        "samples": len(y_true),

        "mae": round(
            float(mae),
            2,
        ),

        "rmse": round(
            float(rmse),
            2,
        ),

        "r2": round(
            float(r2),
            4,
        ),

        "within_10k_pct": round(
            float(within_10k),
            2,
        ),

        "within_20k_pct": round(
            float(within_20k),
            2,
        ),

        "within_40k_pct": round(
            float(within_40k),
            2,
        ),

        "avg_overprediction": round(
            float(avg_overprediction),
            2,
        ),

        "avg_underprediction": round(
            float(avg_underprediction),
            2,
        ),

        "baseline_mae": BASELINE_MAE,
        "baseline_rmse": BASELINE_RMSE,
        "baseline_r2": BASELINE_R2,

        "mae_degradation_pct": round(
            (
                (mae - BASELINE_MAE)
                / BASELINE_MAE
            )
            * 100,
            2,
        ),
    }