import os
from pathlib import Path

from aws_local.storage import (
    download_file,
    object_exists,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "archive.zip"
)

S3_CACHE_PATH = (
    PROJECT_ROOT
    / "tmp_s3_data"
    / "archive.zip"
)

S3_DATA_KEY = os.getenv(
    "S3_DATA_KEY",
    "datasets/raw/archive.zip",
)


def resolve_training_data_path() -> Path:
    data_source = os.getenv(
        "DATA_SOURCE",
        "local",
    ).strip().lower()

    if data_source == "local":
        if not LOCAL_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Local dataset not found: "
                f"{LOCAL_DATA_PATH}"
            )

        return LOCAL_DATA_PATH

    if data_source == "s3":
        if not object_exists(S3_DATA_KEY):
            raise FileNotFoundError(
                f"S3 object not found: "
                f"{S3_DATA_KEY}"
            )

        download_file(
            S3_DATA_KEY,
            str(S3_CACHE_PATH),
        )

        return S3_CACHE_PATH

    raise ValueError(
        "DATA_SOURCE must be either "
        "'local' or 's3'."
    )