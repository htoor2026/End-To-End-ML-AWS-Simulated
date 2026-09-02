from typing import Any

from monitoring.model_promotion import evaluate_candidate
from monitoring.model_registry import get_champion
from monitoring.retraining_orchestrator import (
    run_retraining_workflow,
)
from monitoring.shared_evaluation import (
    evaluate_run_on_production_data,
)


def run_retraining_evaluation() -> dict[str, Any]:
    workflow = run_retraining_workflow()

    if not workflow[
        "candidate_trained"
    ]:
        return {
            "evaluation_started": False,
            "decision": workflow[
                "decision"
            ],
            "workflow": workflow,
        }

    candidate_run_id = workflow[
        "candidate_run_id"
    ]

    champion = get_champion()

    champion_run_id = champion[
        "run_id"
    ]

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
            "evaluation_started": False,
            "decision": (
                "insufficient_evaluation_data"
            ),
            "champion_evaluation": (
                champion_evaluation
            ),
            "workflow": workflow,
        }

    if not candidate_evaluation[
        "evaluated"
    ]:
        return {
            "evaluation_started": False,
            "decision": (
                "insufficient_evaluation_data"
            ),
            "candidate_evaluation": (
                candidate_evaluation
            ),
            "workflow": workflow,
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

    return {
        "evaluation_started": True,
        "candidate_run_id": (
            candidate_run_id
        ),
        "champion_run_id": (
            champion_run_id
        ),
        "champion": champion,
        "evaluation_rows": (
            champion_evaluation[
                "rows"
            ]
        ),
        "promotion_recommended": (
            evaluation["promote"]
        ),
        "evaluation": evaluation,
        "workflow": workflow,
    }


if __name__ == "__main__":
    result = (
        run_retraining_evaluation()
    )

    print(
        "\nRETRAINING EVALUATION RESULT\n"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )