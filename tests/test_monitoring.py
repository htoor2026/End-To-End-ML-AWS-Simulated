import json

from monitoring import drift_monitor, performance_monitor
from monitoring.model_promotion import (
    evaluate_candidate,
)
from monitoring.retraining_policy import (
    evaluate_retraining_policy,
)


def test_candidate_passes_promotion_gates():
    champion = {
        "mae": 13000.0,
        "rmse": 18000.0,
        "r2": 0.92,
    }

    candidate = {
        "mae": 12000.0,
        "rmse": 17500.0,
        "r2": 0.93,
    }

    result = evaluate_candidate(
        candidate_metrics=candidate,
        champion_metrics=champion,
    )

    assert result["promote"] is True

    assert (
        result["decision"]
        == "promote_candidate"
    )

    assert (
        result["gates"][
            "mae_improved_enough"
        ]
        is True
    )

    assert (
        result["gates"][
            "rmse_within_tolerance"
        ]
        is True
    )

    assert (
        result["gates"][
            "r2_within_tolerance"
        ]
        is True
    )


def test_candidate_fails_promotion_gates():
    champion = {
        "mae": 13000.0,
        "rmse": 18000.0,
        "r2": 0.92,
    }

    candidate = {
        "mae": 15000.0,
        "rmse": 20000.0,
        "r2": 0.89,
    }

    result = evaluate_candidate(
        candidate_metrics=candidate,
        champion_metrics=champion,
    )

    assert result["promote"] is False

    assert (
        result["decision"]
        == "reject_candidate"
    )

    assert (
        result["gates"][
            "mae_improved_enough"
        ]
        is False
    )


def test_retraining_policy_waits_for_more_labels(
    monkeypatch,
):
    monkeypatch.setattr(
        (
            "monitoring.retraining_policy."
            "generate_drift_report"
        ),
        lambda: "fake-report",
    )

    monkeypatch.setattr(
        (
            "monitoring.retraining_policy."
            "overall_drift_status"
        ),
        lambda report: "critical",
    )

    monkeypatch.setattr(
        (
            "monitoring.retraining_policy."
            "calculate_performance"
        ),
        lambda: {
            "status": "insufficient_data",
            "samples": 5,
        },
    )

    result = (
        evaluate_retraining_policy()
    )

    assert (
        result[
            "retrain_recommended"
        ]
        is False
    )

    assert (
        result["decision"]
        == "wait_for_more_labels"
    )

    assert (
        result["labelled_samples"]
        == 5
    )


def test_retraining_policy_recommends_candidate_when_degraded(
    monkeypatch,
):
    monkeypatch.setattr(
        (
            "monitoring.retraining_policy."
            "generate_drift_report"
        ),
        lambda: "fake-report",
    )

    monkeypatch.setattr(
        (
            "monitoring.retraining_policy."
            "overall_drift_status"
        ),
        lambda report: "critical",
    )

    monkeypatch.setattr(
        (
            "monitoring.retraining_policy."
            "calculate_performance"
        ),
        lambda: {
            "status": "degraded",
            "samples": 50,
        },
    )

    result = (
        evaluate_retraining_policy()
    )

    assert (
        result[
            "retrain_recommended"
        ]
        is True
    )

    assert (
        result["decision"]
        == "retrain_candidate"
    )

    assert (
        result["labelled_samples"]
        == 50
    )

def test_drift_monitor_uses_latest_100_predictions(
    tmp_path,
    monkeypatch,
):
    log_path = (
        tmp_path
        / "predictions.jsonl"
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index in range(150):
            record = {
                "features": {
                    "Lot Area": index,
                    "Overall Qual": 5,
                    "Year Built": 2000,
                    "Gr Liv Area": 1500,
                    "Bedroom AbvGr": 3,
                    "Garage Cars": 2,
                }
            }

            file.write(
                json.dumps(record)
                + "\n"
            )

    monkeypatch.setattr(
        drift_monitor,
        "PREDICTIONS_LOG_PATH",
        log_path,
    )

    dataframe = (
        drift_monitor
        .load_production_data()
    )

    assert len(dataframe) == 100

    assert (
        dataframe[
            "Lot Area"
        ].iloc[0]
        == 50
    )

    assert (
        dataframe[
            "Lot Area"
        ].iloc[-1]
        == 149
    )

def test_performance_monitor_uses_latest_50_actuals(
    tmp_path,
    monkeypatch,
):
    log_path = (
        tmp_path
        / "actuals.jsonl"
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index in range(80):
            record = {
                "request_id": (
                    f"request-{index}"
                ),
                "predicted_price": (
                    100000 + index
                ),
                "actual_price": (
                    110000 + index
                ),
            }

            file.write(
                json.dumps(record)
                + "\n"
            )

    monkeypatch.setattr(
        performance_monitor,
        "ACTUALS_LOG_PATH",
        log_path,
    )

    dataframe = (
        performance_monitor
        .load_actuals()
    )

    assert len(dataframe) == 50

    assert (
        dataframe[
            "request_id"
        ].iloc[0]
        == "request-30"
    )

    assert (
        dataframe[
            "request_id"
        ].iloc[-1]
        == "request-79"
    )