from typing import Any

from monitoring.candidate_training import train_candidate
from monitoring.retraining_policy import evaluate_retraining_policy


def run_retraining_workflow() -> dict[str, Any]:
    policy = evaluate_retraining_policy()

    if not policy["retrain_recommended"]:
        return {
            "workflow_started": False,
            "candidate_trained": False,
            "decision": policy["decision"],
            "policy": policy,
            "reason": (
                "Retraining policy did not authorize "
                "candidate training."
            ),
        }

    candidate_result = train_candidate()

    if not candidate_result["trained"]:
        return {
            "workflow_started": True,
            "candidate_trained": False,
            "decision": "candidate_training_skipped",
            "policy": policy,
            "candidate": candidate_result,
        }

    return {
        "workflow_started": True,
        "candidate_trained": True,
        "decision": "candidate_created",
        "policy": policy,
        "candidate_run_id": candidate_result[
            "run_id"
        ],
        "candidate": candidate_result,
    }


if __name__ == "__main__":
    result = run_retraining_workflow()

    print(
        "\nRETRAINING WORKFLOW RESULT\n"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )