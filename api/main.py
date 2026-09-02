import logging
import os
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from monitoring.actuals_logger import log_actual
from monitoring.drift_monitor import (
    generate_drift_report,
    overall_drift_status,
)
from monitoring.mlflow_candidate import get_run_metrics
from monitoring.model_promotion import evaluate_candidate
from monitoring.model_registry import (
    get_champion,
    get_or_register_model_version,
    promote_candidate,
    rollback_champion,
)
from monitoring.performance_monitor import calculate_performance
from monitoring.prediction_logger import log_prediction
from monitoring.retraining_lifecycle import (
    run_retraining_lifecycle,
)
from monitoring.retraining_orchestrator import (
    run_retraining_workflow,
)
from monitoring.shared_evaluation import (
    evaluate_run_on_production_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(
    "prices_predictor_api"
)


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    [
        "method",
        "path",
        "status",
    ],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    [
        "method",
        "path",
    ],
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests",
)


app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predict house prices using "
        "the production ML model."
    ),
    version="1.0.0",
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

FRONTEND_DIR = (
    PROJECT_ROOT
    / "frontend"
)


MLFLOW_MODEL_URL = os.getenv(
    "MLFLOW_MODEL_URL",
    "http://127.0.0.1:8000/invocations",
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------


class CandidateRunOnlyInput(
    BaseModel
):
    candidate_run_id: str = Field(
        min_length=1
    )


class ChampionCandidateRunInput(
    BaseModel
):
    champion_run_id: str = Field(
        min_length=1
    )

    candidate_run_id: str = Field(
        min_length=1
    )


class CandidateMetricsInput(
    BaseModel
):
    mae: float = Field(
        gt=0
    )

    rmse: float = Field(
        gt=0
    )

    r2: float


class CandidateRunInput(
    BaseModel
):
    run_id: str = Field(
        min_length=1
    )


class ActualPriceInput(
    BaseModel
):
    request_id: str = Field(
        min_length=1
    )

    actual_price: float = Field(
        gt=0
    )


class HouseFeatures(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True
    )

    order: int = Field(
        gt=0,
        alias="Order",
    )

    pid: int = Field(
        gt=0,
        alias="PID",
    )

    ms_subclass: int = Field(
        gt=0,
        alias="MS SubClass",
    )

    lot_frontage: float = Field(
        ge=0,
        alias="Lot Frontage",
    )

    lot_area: int = Field(
        gt=0,
        alias="Lot Area",
    )

    overall_qual: int = Field(
        ge=1,
        le=10,
        alias="Overall Qual",
    )

    overall_cond: int = Field(
        ge=1,
        le=10,
        alias="Overall Cond",
    )

    year_built: int = Field(
        ge=1800,
        le=2030,
        alias="Year Built",
    )

    year_remod_add: int = Field(
        ge=1800,
        le=2030,
        alias="Year Remod/Add",
    )

    mas_vnr_area: float = Field(
        ge=0,
        alias="Mas Vnr Area",
    )

    bsmtfin_sf_1: float = Field(
        ge=0,
        alias="BsmtFin SF 1",
    )

    bsmtfin_sf_2: float = Field(
        ge=0,
        alias="BsmtFin SF 2",
    )

    bsmt_unf_sf: float = Field(
        ge=0,
        alias="Bsmt Unf SF",
    )

    total_bsmt_sf: float = Field(
        ge=0,
        alias="Total Bsmt SF",
    )

    first_flr_sf: int = Field(
        gt=0,
        alias="1st Flr SF",
    )

    second_flr_sf: int = Field(
        ge=0,
        alias="2nd Flr SF",
    )

    low_qual_fin_sf: int = Field(
        ge=0,
        alias="Low Qual Fin SF",
    )

    gr_liv_area: float = Field(
        gt=0,
        alias="Gr Liv Area",
    )

    bsmt_full_bath: int = Field(
        ge=0,
        alias="Bsmt Full Bath",
    )

    bsmt_half_bath: int = Field(
        ge=0,
        alias="Bsmt Half Bath",
    )

    full_bath: int = Field(
        ge=0,
        alias="Full Bath",
    )

    half_bath: int = Field(
        ge=0,
        alias="Half Bath",
    )

    bedroom_abvgr: int = Field(
        ge=0,
        alias="Bedroom AbvGr",
    )

    kitchen_abvgr: int = Field(
        ge=0,
        alias="Kitchen AbvGr",
    )

    totrms_abvgrd: int = Field(
        gt=0,
        alias="TotRms AbvGrd",
    )

    fireplaces: int = Field(
        ge=0,
        alias="Fireplaces",
    )

    garage_yr_blt: int = Field(
        ge=1800,
        le=2030,
        alias="Garage Yr Blt",
    )

    garage_cars: int = Field(
        ge=0,
        alias="Garage Cars",
    )

    garage_area: float = Field(
        ge=0,
        alias="Garage Area",
    )

    wood_deck_sf: float = Field(
        ge=0,
        alias="Wood Deck SF",
    )

    open_porch_sf: int = Field(
        ge=0,
        alias="Open Porch SF",
    )

    enclosed_porch: int = Field(
        ge=0,
        alias="Enclosed Porch",
    )

    three_ssn_porch: int = Field(
        ge=0,
        alias="3Ssn Porch",
    )

    screen_porch: int = Field(
        ge=0,
        alias="Screen Porch",
    )

    pool_area: int = Field(
        ge=0,
        alias="Pool Area",
    )

    misc_val: int = Field(
        ge=0,
        alias="Misc Val",
    )

    mo_sold: int = Field(
        ge=1,
        le=12,
        alias="Mo Sold",
    )

    yr_sold: int = Field(
        ge=1900,
        le=2030,
        alias="Yr Sold",
    )


# ---------------------------------------------------------
# Middleware
# ---------------------------------------------------------


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = (
        request_id
    )

    start_time = (
        time.perf_counter()
    )

    logger.info(
        (
            "request_started | "
            "request_id=%s | "
            "method=%s | "
            "path=%s"
        ),
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(
            request
        )

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        latency_ms = (
            latency_seconds
            * 1000
        )

        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=(
                response.status_code
            ),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            path=request.url.path,
        ).observe(
            latency_seconds
        )

        logger.info(
            (
                "request_completed | "
                "request_id=%s | "
                "method=%s | "
                "path=%s | "
                "status=%s | "
                "latency_ms=%.2f"
            ),
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    except Exception:
        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        latency_ms = (
            latency_seconds
            * 1000
        )

        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=500,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            path=request.url.path,
        ).observe(
            latency_seconds
        )

        logger.exception(
            (
                "request_failed | "
                "request_id=%s | "
                "method=%s | "
                "path=%s | "
                "latency_ms=%.2f"
            ),
            request_id,
            request.method,
            request.url.path,
            latency_ms,
        )

        raise


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------


app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="static",
)

@app.post("/retraining/lifecycle")
def retraining_lifecycle():
    return run_retraining_lifecycle()

@app.get("/app")
def frontend():
    return FileResponse(
        FRONTEND_DIR
        / "index.html"
    )

@app.post("/retraining/run")
def run_retraining():
    result = run_retraining_workflow()

    return result

# ---------------------------------------------------------
# Basic API endpoints
# ---------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": (
            "House Price Prediction API"
        ),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": (
            "prices_predictor"
        ),
        "task": "regression",
        "target": "SalePrice",
        "status": "active",
    }


# ---------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=(
            CONTENT_TYPE_LATEST
        ),
    )


@app.get("/drift")
def drift():
    report = (
        generate_drift_report()
    )

    return {
        "overall_status": (
            overall_drift_status(
                report
            )
        ),
        "rows": report.to_dict(
            orient="records"
        ),
    }


@app.get("/performance")
def performance():
    return (
        calculate_performance()
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------


@app.post("/predict")
def predict(
    features: HouseFeatures,
    request: Request,
):
    PREDICTION_COUNT.inc()

    house_data = (
        features.model_dump(
            by_alias=True
        )
    )

    payload = {
        "dataframe_records": [
            house_data
        ]
    }

    try:
        response = requests.post(
            MLFLOW_MODEL_URL,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        if (
            not result
            or "predictions"
            not in result
            or len(
                result["predictions"]
            )
            == 0
        ):
            PREDICTION_ERROR_COUNT.inc()

            raise HTTPException(
                status_code=502,
                detail=(
                    "Invalid response "
                    "received from MLflow "
                    "model server."
                ),
            )

        prediction = float(
            result[
                "predictions"
            ][0]
        )

        log_prediction(
            request_id=(
                request.state.request_id
            ),
            features=house_data,
            prediction=prediction,
        )

        return {
            "status": "success",
            "predicted_price": round(
                prediction,
                2,
            ),
            "currency": "USD",
        }

    except (
        requests.exceptions.ConnectionError
    ) as error:
        PREDICTION_ERROR_COUNT.inc()

        raise HTTPException(
            status_code=503,
            detail=(
                "MLflow model server "
                "is unavailable."
            ),
        ) from error

    except (
        requests.exceptions.Timeout
    ) as error:
        PREDICTION_ERROR_COUNT.inc()

        raise HTTPException(
            status_code=504,
            detail=(
                "Model prediction "
                "timed out."
            ),
        ) from error

    except (
        requests.exceptions.RequestException
    ) as error:
        PREDICTION_ERROR_COUNT.inc()

        raise HTTPException(
            status_code=502,
            detail=(
                "Prediction service error: "
                f"{error!s}"
            ),
        ) from error


# ---------------------------------------------------------
# Actual labels
# ---------------------------------------------------------


@app.post("/actual")
def actual(
    actual_data: ActualPriceInput,
):
    try:
        record = log_actual(
            request_id=(
                actual_data.request_id
            ),
            actual_price=(
                actual_data.actual_price
            ),
        )

        return {
            "status": "success",
            "record": record,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


# ---------------------------------------------------------
# Manual promotion diagnostics
# ---------------------------------------------------------


@app.post("/promotion-check")
def promotion_check(
    candidate: CandidateMetricsInput,
):
    """
    Diagnostic endpoint.

    Candidate metrics are manually provided.
    Champion metrics are read from the current
    champion MLflow run.
    """

    champion = get_champion()

    champion_metrics = (
        get_run_metrics(
            champion[
                "run_id"
            ]
        )
    )

    return evaluate_candidate(
        candidate_metrics={
            "mae": candidate.mae,
            "rmse": candidate.rmse,
            "r2": candidate.r2,
        },
        champion_metrics=(
            champion_metrics
        ),
    )


@app.post(
    "/promotion-check/mlflow"
)
def promotion_check_mlflow(
    candidate_run: CandidateRunInput,
):
    """
    Diagnostic endpoint.

    Compare candidate MLflow metrics
    against the current champion run's
    original MLflow metrics.
    """

    try:
        champion = (
            get_champion()
        )

        champion_metrics = (
            get_run_metrics(
                champion[
                    "run_id"
                ]
            )
        )

        candidate_metrics = (
            get_run_metrics(
                candidate_run.run_id
            )
        )

        return evaluate_candidate(
            candidate_metrics=(
                candidate_metrics
            ),
            champion_metrics=(
                champion_metrics
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/promotion-check/mlflow-runs"
)
def promotion_check_mlflow_runs(
    run_input: (
        ChampionCandidateRunInput
    ),
):
    """
    Diagnostic endpoint.

    Compare any two explicitly supplied
    MLflow runs using their stored metrics.
    """

    try:
        champion_metrics = (
            get_run_metrics(
                run_input.champion_run_id
            )
        )

        candidate_metrics = (
            get_run_metrics(
                run_input.candidate_run_id
            )
        )

        return evaluate_candidate(
            candidate_metrics=(
                candidate_metrics
            ),
            champion_metrics=(
                champion_metrics
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ---------------------------------------------------------
# Production-quality shared evaluation
# ---------------------------------------------------------


@app.post(
    "/promotion-check/registry"
)
def promotion_check_registry(
    run_input: CandidateRunOnlyInput,
):
    """
    Production promotion check.

    Champion and candidate are evaluated
    on the same labelled production data.
    """

    champion = get_champion()

    champion_run_id = (
        champion[
            "run_id"
        ]
    )

    candidate_run_id = (
        run_input.candidate_run_id
    )

    champion_evaluation = (
        evaluate_run_on_production_data(
            champion_run_id
        )
    )

    candidate_evaluation = (
        evaluate_run_on_production_data(
            candidate_run_id
        )
    )

    if not champion_evaluation[
        "evaluated"
    ]:
        return {
            "promote": False,
            "decision": (
                "insufficient_evaluation_data"
            ),
            "champion": champion,
            "champion_evaluation": (
                champion_evaluation
            ),
        }

    if not candidate_evaluation[
        "evaluated"
    ]:
        return {
            "promote": False,
            "decision": (
                "insufficient_evaluation_data"
            ),
            "candidate_run_id": (
                candidate_run_id
            ),
            "candidate_evaluation": (
                candidate_evaluation
            ),
        }

    result = evaluate_candidate(
        candidate_metrics=(
            candidate_evaluation[
                "metrics"
            ]
        ),
        champion_metrics=(
            champion_evaluation[
                "metrics"
            ]
        ),
    )

    return {
        "champion": champion,
        "candidate_run_id": (
            candidate_run_id
        ),
        "evaluation_rows": (
            champion_evaluation[
                "rows"
            ]
        ),
        **result,
    }


# ---------------------------------------------------------
# Actual registry promotion
# ---------------------------------------------------------


@app.post("/promotion/registry")
def promote_registry_candidate(
    run_input: CandidateRunOnlyInput,
):
    """
    Promote a candidate only if it beats
    the current champion on the same
    labelled production evaluation data.
    """

    champion = get_champion()

    champion_run_id = (
        champion[
            "run_id"
        ]
    )

    candidate_run_id = (
        run_input.candidate_run_id
    )

    if (
        candidate_run_id
        == champion_run_id
    ):
        return {
            "promoted": False,
            "decision": (
                "already_champion"
            ),
            "champion": champion,
        }

    champion_evaluation = (
        evaluate_run_on_production_data(
            champion_run_id
        )
    )

    candidate_evaluation = (
        evaluate_run_on_production_data(
            candidate_run_id
        )
    )

    if not champion_evaluation[
        "evaluated"
    ]:
        return {
            "promoted": False,
            "decision": (
                "insufficient_evaluation_data"
            ),
            "champion_evaluation": (
                champion_evaluation
            ),
        }

    if not candidate_evaluation[
        "evaluated"
    ]:
        return {
            "promoted": False,
            "decision": (
                "insufficient_evaluation_data"
            ),
            "candidate_evaluation": (
                candidate_evaluation
            ),
        }

    evaluation = evaluate_candidate(
        candidate_metrics=(
            candidate_evaluation[
                "metrics"
            ]
        ),
        champion_metrics=(
            champion_evaluation[
                "metrics"
            ]
        ),
    )

    if not evaluation[
        "promote"
    ]:
        return {
            "promoted": False,
            **evaluation,
        }

    candidate_version = (
        get_or_register_model_version(
            candidate_run_id
        )
    )

    promotion = promote_candidate(
        candidate_version[
            "version"
        ]
    )

    return {
        "promoted": True,
        "evaluation": evaluation,
        "registry": promotion,
    }


# ---------------------------------------------------------
# Rollback
# ---------------------------------------------------------


@app.post("/rollback/registry")
def rollback_registry_champion():
    result = (
        rollback_champion()
    )

    return {
        "rolled_back": True,
        **result,
    }