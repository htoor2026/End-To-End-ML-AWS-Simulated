from typing import Any

from monitoring.model_registry import (
    get_or_register_model_version,
    promote_candidate,
)
from monitoring.retraining_evaluator import (
    run_retraining_evaluation,
)


def run_retraining_lifecycle() -> dict[str, Any]:
    evaluation_result = (
        run_retraining_evaluation()
    )

    if not evaluation_result.get(
        "evaluation_started",
        False,
    ):
        return {
            "lifecycle_completed": False,
            "promoted": False,
            "decision": evaluation_result[
                "decision"
            ],
            "evaluation_result": (
                evaluation_result
            ),
        }

    promotion_recommended = (
        evaluation_result.get(
            "promotion_recommended",
            False,
        )
    )

    if not promotion_recommended:
        return {
            "lifecycle_completed": True,
            "promoted": False,
            "decision": "candidate_rejected",
            "evaluation_result": (
                evaluation_result
            ),
        }

    candidate_run_id = (
        evaluation_result[
            "candidate_run_id"
        ]
    )

    candidate_version = (
        get_or_register_model_version(
            candidate_run_id
        )
    )

    promotion_result = (
        promote_candidate(
            candidate_version[
                "version"
            ]
        )
    )

    return {
        "lifecycle_completed": True,
        "promoted": True,
        "decision": "candidate_promoted",
        "candidate_run_id": (
            candidate_run_id
        ),
        "candidate_version": (
            candidate_version
        ),
        "promotion_result": (
            promotion_result
        ),
        "evaluation_result": (
            evaluation_result
        ),
    }


if __name__ == "__main__":
    result = (
        run_retraining_lifecycle()
    )

    print(
        "\nRETRAINING LIFECYCLE RESULT\n"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )