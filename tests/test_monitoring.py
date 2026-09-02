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