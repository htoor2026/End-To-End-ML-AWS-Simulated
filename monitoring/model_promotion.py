from typing import Any

# Candidate must improve MAE by at least 1%.
MIN_MAE_IMPROVEMENT_PCT = 1.0

# Candidate cannot materially worsen RMSE.
MAX_RMSE_DEGRADATION_PCT = 2.0

# Candidate cannot lose more than this amount of R².
MAX_R2_DROP = 0.005


def percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    return (
        (new_value - old_value)
        / old_value
    ) * 100


def evaluate_candidate(
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float],
) -> dict[str, Any]:
    required_metrics = {
        "mae",
        "rmse",
        "r2",
    }

    missing_candidate_metrics = (
        required_metrics
        - candidate_metrics.keys()
    )

    missing_champion_metrics = (
        required_metrics
        - champion_metrics.keys()
    )

    if missing_candidate_metrics:
        return {
            "promote": False,
            "decision": "reject_candidate",
            "reasons": [
                (
                    "Candidate metrics are incomplete. "
                    f"Missing: {sorted(missing_candidate_metrics)}"
                )
            ],
        }

    if missing_champion_metrics:
        return {
            "promote": False,
            "decision": "reject_candidate",
            "reasons": [
                (
                    "Champion metrics are incomplete. "
                    f"Missing: {sorted(missing_champion_metrics)}"
                )
            ],
        }

    candidate_mae = candidate_metrics[
        "mae"
    ]

    candidate_rmse = candidate_metrics[
        "rmse"
    ]

    candidate_r2 = candidate_metrics[
        "r2"
    ]

    champion_mae = champion_metrics[
        "mae"
    ]

    champion_rmse = champion_metrics[
        "rmse"
    ]

    champion_r2 = champion_metrics[
        "r2"
    ]

    # MAE is better when lower.
    mae_improvement_pct = (
        (
            champion_mae
            - candidate_mae
        )
        / champion_mae
    ) * 100

    # Positive means candidate RMSE became worse.
    # Negative means candidate RMSE improved.
    rmse_change_pct = percentage_change(
        champion_rmse,
        candidate_rmse,
    )

    # Positive means candidate R² dropped.
    # Negative means candidate R² improved.
    r2_drop = (
        champion_r2
        - candidate_r2
    )

    gates = {
        "mae_improved_enough": (
            mae_improvement_pct
            >= MIN_MAE_IMPROVEMENT_PCT
        ),
        "rmse_within_tolerance": (
            rmse_change_pct
            <= MAX_RMSE_DEGRADATION_PCT
        ),
        "r2_within_tolerance": (
            r2_drop
            <= MAX_R2_DROP
        ),
    }

    reasons = []

    if not gates[
        "mae_improved_enough"
    ]:
        reasons.append(
            
                "Candidate MAE did not improve enough "
                "over the champion."
            
        )

    if not gates[
        "rmse_within_tolerance"
    ]:
        reasons.append(
            
                "Candidate RMSE degraded beyond "
                "the allowed tolerance."
            
        )

    if not gates[
        "r2_within_tolerance"
    ]:
        reasons.append(
            
                "Candidate R² dropped beyond "
                "the allowed tolerance."
            
        )

    promote = all(
        gates.values()
    )

    if promote:
        decision = "promote_candidate"

        reasons.append(
            
                "Candidate passed all model "
                "promotion quality gates."
            
        )

    else:
        decision = "reject_candidate"

    return {
        "promote": promote,
        "decision": decision,
        "champion_metrics": (
            champion_metrics
        ),
        "candidate_metrics": (
            candidate_metrics
        ),
        "comparison": {
            "mae_improvement_pct": round(
                mae_improvement_pct,
                2,
            ),
            "rmse_change_pct": round(
                rmse_change_pct,
                2,
            ),
            "r2_drop": round(
                r2_drop,
                4,
            ),
        },
        "gates": gates,
        "reasons": reasons,
    }


if __name__ == "__main__":
    # Simulated champion metrics.
    champion = {
        "mae": 13044.81,
        "rmse": 18626.84,
        "r2": 0.9221,
    }

    # Simulated candidate metrics.
    candidate = {
        "mae": 12500.0,
        "rmse": 18000.0,
        "r2": 0.9280,
    }

    result = evaluate_candidate(
        candidate_metrics=candidate,
        champion_metrics=champion,
    )

    print(
        "\nMODEL PROMOTION DECISION\n"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )