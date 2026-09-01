import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using the production ML model.",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRONTEND_DIR = PROJECT_ROOT / "frontend"


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

@app.get("/app")
def frontend():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )

import os

MLFLOW_MODEL_URL = os.getenv(
    "MLFLOW_MODEL_URL",
    "http://127.0.0.1:8000/invocations",
)


class HouseFeatures(BaseModel):
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


@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
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
        "model_name": "prices_predictor",
        "task": "regression",
        "target": "SalePrice",
        "status": "active",
    }


@app.post("/predict")
def predict(features: HouseFeatures):

    # Convert our clean FastAPI input names back to
    # the exact column names expected by the MLflow model
    house_data = features.model_dump(by_alias=True)

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
            or "predictions" not in result
            or len(result["predictions"]) == 0
        ):
            raise HTTPException(
                status_code=502,
                detail="Invalid response received from MLflow model server.",
            )

        prediction = float(
            result["predictions"][0]
        )

        return {
            "status": "success",
            "predicted_price": round(
                prediction,
                2,
            ),
            "currency": "USD",
        }

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="MLflow model server is unavailable.",
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Model prediction timed out.",
        )

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Prediction service error: {str(error)}",
        )