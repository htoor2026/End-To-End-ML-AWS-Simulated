import os
from urllib.parse import urlparse

AWS_ENDPOINT_URL = os.getenv(
    "AWS_ENDPOINT_URL",
    "http://127.0.0.1:5000",
)

AWS_REGION = os.getenv(
    "AWS_DEFAULT_REGION",
    "us-east-1",
)

AWS_ACCESS_KEY_ID = os.getenv(
    "AWS_ACCESS_KEY_ID",
    "test",
)

AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY",
    "test",
)

DATA_BUCKET = os.getenv(
    "AWS_DATA_BUCKET",
    "ml-local-data",
)


def validate_local_aws_endpoint() -> None:
    parsed_url = urlparse(
        AWS_ENDPOINT_URL
    )

    allowed_hosts = {
        "127.0.0.1",
        "localhost",
    }

    if parsed_url.hostname not in allowed_hosts:
        raise RuntimeError(
            "Blocked AWS endpoint. "
            "This project only allows local AWS simulation."
        )