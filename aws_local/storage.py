from pathlib import Path

from botocore.exceptions import ClientError

from aws_local.config import DATA_BUCKET
from aws_local.s3_client import get_s3_client


def upload_file(
    local_path: str,
    object_key: str,
) -> None:
    s3 = get_s3_client()

    s3.upload_file(
        local_path,
        DATA_BUCKET,
        object_key,
    )


def download_file(
    object_key: str,
    destination_path: str,
) -> None:
    s3 = get_s3_client()

    destination = Path(
        destination_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3.download_file(
        DATA_BUCKET,
        object_key,
        str(destination),
    )


def object_exists(
    object_key: str,
) -> bool:
    s3 = get_s3_client()

    try:
        s3.head_object(
            Bucket=DATA_BUCKET,
            Key=object_key,
        )

        return True

    except ClientError as error:
        error_code = error.response[
            "Error"
        ]["Code"]

        if error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise