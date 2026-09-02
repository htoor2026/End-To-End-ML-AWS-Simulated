from monitoring.drift_monitor import (
    generate_drift_report,
    overall_drift_status,
)
from monitoring.performance_monitor import (
    calculate_performance,
)

MIN_LABELLED_SAMPLES_FOR_RETRAINING = 30


def evaluate_retraining_policy() -> dict:
    drift_report = generate_drift_report()

    drift_status = overall_drift_status(
        drift_report
    )

    performance = calculate_performance()

    performance_status = performance[
        "status"
    ]

    labelled_samples = performance[
        "samples"
    ]

    reasons = []

    # Do not make a retraining decision
    # without enough labelled production outcomes.
    if (
        labelled_samples
        < MIN_LABELLED_SAMPLES_FOR_RETRAINING
    ):
        return {
            "retrain_recommended": False,
            "decision": "wait_for_more_labels",
            "labelled_samples": labelled_samples,
            "minimum_labelled_samples_required": (
                MIN_LABELLED_SAMPLES_FOR_RETRAINING
            ),
            "performance_status": (
                performance_status
            ),
            "drift_status": drift_status,
            "reasons": [
                (
                    "Not enough labelled production data "
                    "to make a reliable retraining decision."
                )
            ],
        }

    # Actual performance degradation is the
    # primary signal for retraining.
    if performance_status == "degraded":
        reasons.append(
            
                "Production model performance has degraded "
                "beyond the allowed threshold."
            
        )

    # Drift is supporting evidence.
    if drift_status == "critical":
        reasons.append(
            "Significant input data drift detected."
        )

    elif drift_status == "warning":
        reasons.append(
            "Moderate input data drift detected."
        )

    # Do not retrain based only on drift.
    retrain_recommended = (
        performance_status == "degraded"
    )

    if retrain_recommended:
        decision = "retrain_candidate"
    else:
        decision = "continue_monitoring"

    if not reasons:
        reasons.append(
            
                "Production performance remains within "
                "the accepted threshold."
            
        )

    return {
        "retrain_recommended": (
            retrain_recommended
        ),
        "decision": decision,
        "labelled_samples": labelled_samples,
        "minimum_labelled_samples_required": (
            MIN_LABELLED_SAMPLES_FOR_RETRAINING
        ),
        "performance_status": (
            performance_status
        ),
        "drift_status": drift_status,
        "reasons": reasons,
    }


if __name__ == "__main__":
    policy_result = (
        evaluate_retraining_policy()
    )

    print(
        "\nRETRAINING POLICY DECISION\n"
    )

    for key, value in policy_result.items():
        print(
            f"{key}: {value}"
        )