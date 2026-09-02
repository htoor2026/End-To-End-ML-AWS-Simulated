import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "prices_predictor"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)


def configure_mlflow() -> None:
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )


def register_model_version(
    run_id: str,
) -> dict:
    configure_mlflow()

    client = MlflowClient()

    try:
        client.get_registered_model(
            MODEL_NAME
        )

    except MlflowException:
        client.create_registered_model(
            MODEL_NAME
        )

    model_uri = (
        f"runs:/{run_id}/model"
    )

    version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    return {
        "model_name": MODEL_NAME,
        "version": version.version,
        "run_id": run_id,
    }


def set_champion(
    version: str,
) -> dict:
    configure_mlflow()

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )

    model_version = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    )

    return {
        "model_name": MODEL_NAME,
        "alias": "champion",
        "version": model_version.version,
        "run_id": model_version.run_id,
    }


def get_champion() -> dict:
    configure_mlflow()

    client = MlflowClient()

    model_version = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    )

    return {
        "model_name": MODEL_NAME,
        "alias": "champion",
        "version": model_version.version,
        "run_id": model_version.run_id,
    }
def get_or_register_model_version(
    run_id: str,
) -> dict:
    configure_mlflow()

    client = MlflowClient()

    versions = client.search_model_versions(
        filter_string=(
            f"name='{MODEL_NAME}'"
        )
    )

    for version in versions:
        if version.run_id == run_id:
            return {
                "model_name": MODEL_NAME,
                "version": version.version,
                "run_id": version.run_id,
                "already_registered": True,
            }

    registered = register_model_version(
        run_id
    )

    return {
        **registered,
        "already_registered": False,
    }


def promote_candidate(
    candidate_version: str,
) -> dict:
    configure_mlflow()

    client = MlflowClient()

    current_champion = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    )

    # Preserve the existing champion
    # so we can roll back later.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="previous_champion",
        version=current_champion.version,
    )

    # Promote the candidate.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=candidate_version,
    )

    new_champion = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    )

    return {
        "status": "promoted",
        "model_name": MODEL_NAME,
        "previous_champion": {
            "version": (
                current_champion.version
            ),
            "run_id": (
                current_champion.run_id
            ),
        },
        "new_champion": {
            "version": (
                new_champion.version
            ),
            "run_id": (
                new_champion.run_id
            ),
        },
    }


def rollback_champion() -> dict:
    configure_mlflow()

    client = MlflowClient()

    current_champion = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    )

    previous_champion = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="previous_champion",
        )
    )

    # Move champion back to the previous version.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=previous_champion.version,
    )

    # Keep the model we rolled back from
    # available for audit/debugging.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="rolled_back",
        version=current_champion.version,
    )

    restored_champion = (
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    )

    return {
        "status": "rolled_back",
        "model_name": MODEL_NAME,
        "rolled_back_from": {
            "version": (
                current_champion.version
            ),
            "run_id": (
                current_champion.run_id
            ),
        },
        "restored_champion": {
            "version": (
                restored_champion.version
            ),
            "run_id": (
                restored_champion.run_id
            ),
        },
    }