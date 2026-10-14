import boto3
import pytest
from moto import mock_aws

import aws_local.config as aws_config
from aws_local.config import (
    validate_local_aws_endpoint,
)


def test_local_aws_endpoint_is_allowed():
    validate_local_aws_endpoint()

@mock_aws
def test_simulated_s3_upload_and_download():
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    bucket = "test-ml-bucket"

    s3.create_bucket(
        Bucket=bucket
    )

    s3.put_object(
        Bucket=bucket,
        Key="datasets/test.txt",
        Body=b"hello ml",
    )

    response = s3.get_object(
        Bucket=bucket,
        Key="datasets/test.txt",
    )

    content = response[
        "Body"
    ].read()

    assert content == b"hello ml"

def test_real_aws_endpoint_is_blocked(
    monkeypatch,
):
    monkeypatch.setattr(
        aws_config,
        "AWS_ENDPOINT_URL",
        "https://s3.amazonaws.com",
    )

    with pytest.raises(
        RuntimeError,
        match="Blocked AWS endpoint",
    ):
        aws_config.validate_local_aws_endpoint()