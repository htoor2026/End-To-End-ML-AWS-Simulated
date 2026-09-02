from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


VALID_HOUSE = {
    "Order": 1,
    "PID": 5286,
    "MS SubClass": 20,
    "Lot Frontage": 80.0,
    "Lot Area": 9600,
    "Overall Qual": 5,
    "Overall Cond": 7,
    "Year Built": 1961,
    "Year Remod/Add": 1961,
    "Mas Vnr Area": 0.0,
    "BsmtFin SF 1": 700.0,
    "BsmtFin SF 2": 0.0,
    "Bsmt Unf SF": 150.0,
    "Total Bsmt SF": 850.0,
    "1st Flr SF": 856,
    "2nd Flr SF": 854,
    "Low Qual Fin SF": 0,
    "Gr Liv Area": 1710.0,
    "Bsmt Full Bath": 1,
    "Bsmt Half Bath": 0,
    "Full Bath": 1,
    "Half Bath": 0,
    "Bedroom AbvGr": 3,
    "Kitchen AbvGr": 1,
    "TotRms AbvGrd": 7,
    "Fireplaces": 2,
    "Garage Yr Blt": 1961,
    "Garage Cars": 2,
    "Garage Area": 500.0,
    "Wood Deck SF": 210.0,
    "Open Porch SF": 0,
    "Enclosed Porch": 0,
    "3Ssn Porch": 0,
    "Screen Porch": 0,
    "Pool Area": 0,
    "Misc Val": 0,
    "Mo Sold": 5,
    "Yr Sold": 2010,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "House Price Prediction API"
    assert data["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "prices_predictor"
    assert data["task"] == "regression"
    assert data["target"] == "SalePrice"
    assert data["status"] == "active"


@patch("api.main.requests.post")
def test_valid_prediction(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "predictions": [190330.70931474073]
    }

    mock_post.return_value.raise_for_status.return_value = None

    response = client.post(
        "/predict",
        json=VALID_HOUSE,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["predicted_price"] == 190330.71
    assert data["currency"] == "USD"

    assert isinstance(
        data["predicted_price"],
        (int, float),
    )


def test_invalid_overall_quality():
    invalid_house = VALID_HOUSE.copy()

    invalid_house["Overall Qual"] = 0

    response = client.post(
        "/predict",
        json=invalid_house,
    )

    assert response.status_code == 422


def test_overall_quality_above_maximum():
    invalid_house = VALID_HOUSE.copy()

    invalid_house["Overall Qual"] = 11

    response = client.post(
        "/predict",
        json=invalid_house,
    )

    assert response.status_code == 422


def test_invalid_year_built():
    invalid_house = VALID_HOUSE.copy()

    invalid_house["Year Built"] = 1200

    response = client.post(
        "/predict",
        json=invalid_house,
    )

    assert response.status_code == 422


def test_invalid_month():
    invalid_house = VALID_HOUSE.copy()

    invalid_house["Mo Sold"] = 13

    response = client.post(
        "/predict",
        json=invalid_house,
    )

    assert response.status_code == 422


def test_zero_living_area():
    invalid_house = VALID_HOUSE.copy()

    invalid_house["Gr Liv Area"] = 0

    response = client.post(
        "/predict",
        json=invalid_house,
    )

    assert response.status_code == 422


def test_missing_required_feature():
    incomplete_house = VALID_HOUSE.copy()

    del incomplete_house["Gr Liv Area"]

    response = client.post(
        "/predict",
        json=incomplete_house,
    )

    assert response.status_code == 422


def test_wrong_data_type():
    invalid_house = VALID_HOUSE.copy()

    invalid_house["Overall Qual"] = "excellent"

    response = client.post(
        "/predict",
        json=invalid_house,
    )

    assert response.status_code == 422


@patch("api.main.requests.post")
def test_mlflow_server_unavailable(mock_post):
    mock_post.side_effect = (
        requests.exceptions.ConnectionError
    )

    response = client.post(
        "/predict",
        json=VALID_HOUSE,
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "MLflow model server is unavailable."
    }


@patch("api.main.requests.post")
def test_mlflow_timeout(mock_post):
    mock_post.side_effect = (
        requests.exceptions.Timeout
    )

    response = client.post(
        "/predict",
        json=VALID_HOUSE,
    )

    assert response.status_code == 504

    assert response.json() == {
        "detail": "Model prediction timed out."
    }


@patch("api.main.requests.post")
def test_prediction_request_sent_correctly(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "predictions": [190330.70931474073]
    }

    mock_post.return_value.raise_for_status.return_value = None

    response = client.post(
        "/predict",
        json=VALID_HOUSE,
    )

    assert response.status_code == 200

    mock_post.assert_called_once()

    call_kwargs = mock_post.call_args.kwargs

    assert call_kwargs["json"] == {
        "dataframe_records": [
            VALID_HOUSE
        ]
    }

    assert call_kwargs["timeout"] == 30


@patch("api.main.requests.post")
def test_prediction_is_positive_for_valid_example(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "predictions": [190330.70931474073]
    }

    mock_post.return_value.raise_for_status.return_value = None

    response = client.post(
        "/predict",
        json=VALID_HOUSE,
    )

    prediction = response.json()[
        "predicted_price"
    ]

    assert prediction > 0

def test_retraining_lifecycle_waits_for_more_labels(
    monkeypatch,
):
    mock_result = {
        "lifecycle_completed": False,
        "promoted": False,
        "decision": "wait_for_more_labels",
        "evaluation_result": {
            "evaluation_started": False,
            "decision": "wait_for_more_labels",
        },
    }

    monkeypatch.setattr(
        "api.main.run_retraining_lifecycle",
        lambda: mock_result,
    )

    response = client.post(
        "/retraining/lifecycle"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["lifecycle_completed"]
        is False
    )

    assert (
        data["promoted"]
        is False
    )

    assert (
        data["decision"]
        == "wait_for_more_labels"
    )

def test_registry_promotion_rejects_bad_candidate(
    monkeypatch,
):
    champion = {
        "model_name": "prices_predictor",
        "alias": "champion",
        "version": "1",
        "run_id": "champion-run",
    }

    champion_evaluation = {
        "evaluated": True,
        "run_id": "champion-run",
        "rows": 50,
        "metrics": {
            "mae": 13000.0,
            "rmse": 18000.0,
            "r2": 0.92,
        },
    }

    candidate_evaluation = {
        "evaluated": True,
        "run_id": "candidate-run",
        "rows": 50,
        "metrics": {
            "mae": 14000.0,
            "rmse": 20000.0,
            "r2": 0.90,
        },
    }

    monkeypatch.setattr(
        "api.main.get_champion",
        lambda: champion,
    )

    def mock_evaluation(run_id):
        if run_id == "champion-run":
            return champion_evaluation

        return candidate_evaluation

    monkeypatch.setattr(
        "api.main.evaluate_run_on_production_data",
        mock_evaluation,
    )

    response = client.post(
        "/promotion/registry",
        json={
            "candidate_run_id": "candidate-run"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["promoted"] is False
    assert data["decision"] == "reject_candidate"

    assert (
        data["gates"][
            "mae_improved_enough"
        ]
        is False
    )

def test_registry_promotion_blocks_insufficient_evaluation_data(
    monkeypatch,
):
    champion = {
        "model_name": "prices_predictor",
        "alias": "champion",
        "version": "1",
        "run_id": "champion-run",
    }

    insufficient_evaluation = {
        "evaluated": False,
        "reason": (
            "Not enough labelled production data "
            "for shared model evaluation."
        ),
        "rows": 1,
        "minimum_required": 10,
    }

    monkeypatch.setattr(
        "api.main.get_champion",
        lambda: champion,
    )

    monkeypatch.setattr(
        "api.main.evaluate_run_on_production_data",
        lambda run_id: insufficient_evaluation,
    )

    response = client.post(
        "/promotion/registry",
        json={
            "candidate_run_id": "candidate-run"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["promoted"] is False

    assert (
        data["decision"]
        == "insufficient_evaluation_data"
    )

    assert (
        data["champion_evaluation"][
            "evaluated"
        ]
        is False
    )

    assert (
        data["champion_evaluation"][
            "rows"
        ]
        == 1
    )