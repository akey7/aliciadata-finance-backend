"""
Tests for aliciadata_finance_backend.storage.s3_uploader.

boto3 is replaced with an in-memory fake, load_dotenv is disabled, and the
environment is set with monkeypatch, so no network call is made and the real
.env file is never read.
"""

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from aliciadata_finance_backend.storage import s3_uploader as s3_module
from aliciadata_finance_backend.storage.s3_uploader import (
    BUCKET_NAME_ENV,
    S3Uploader,
    validate_endpoint,
)

SPACE = "test-space"
REGION_ONLY_ENDPOINT = "https://nyc3.digitaloceanspaces.com"


class FakeClient:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, local_filename, bucket_name, remote_filename):
        self.calls.append((local_filename, bucket_name, remote_filename))
        if self.error is not None:
            raise self.error


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(s3_module, "load_dotenv", lambda: None)
    monkeypatch.setenv(BUCKET_NAME_ENV, SPACE)
    monkeypatch.setenv("FSF_FRONT_END_BUCKET_ENDPOINT", REGION_ONLY_ENDPOINT)
    monkeypatch.setenv("FSF_FRONT_END_BUCKET_REGION", "nyc3")
    monkeypatch.setenv("FSF_FRONT_END_BUCKET_KEY_ID", "test-key-id")
    monkeypatch.setenv("FSF_FRONT_END_BUCKET_RWDELETE", "test-secret")


@pytest.fixture
def fake_client(env, monkeypatch):
    """Patch boto3 so S3Uploader gets a FakeClient; swap holder['client'] for errors."""
    holder = {"client": FakeClient()}

    class FakeSession:
        def client(self, *args, **kwargs):
            holder["kwargs"] = kwargs
            return holder["client"]

    monkeypatch.setattr(s3_module.boto3.session, "Session", FakeSession)
    return holder


# Uploading


def test_upload_uses_space_as_bucket_and_full_key(fake_client, capsys):
    S3Uploader().upload_file("local.csv", "time-series/2026-10-12/all_forecasts.csv")
    assert fake_client["client"].calls == [
        ("local.csv", SPACE, "time-series/2026-10-12/all_forecasts.csv")
    ]
    assert fake_client["kwargs"]["endpoint_url"] == REGION_ONLY_ENDPOINT
    assert "time-series/2026-10-12/all_forecasts.csv successful" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing.csv"),
        NoCredentialsError(),
        ClientError({"Error": {"Code": "NoSuchBucket", "Message": "nope"}}, "PutObject"),
    ],
    ids=["file-not-found", "no-credentials", "client-error"],
)
def test_upload_errors_propagate_without_success_message(fake_client, capsys, error):
    fake_client["client"] = FakeClient(error=error)
    with pytest.raises(type(error)):
        S3Uploader().upload_file("local.csv", "folder/2026-10-12/file.csv")
    assert "successful" not in capsys.readouterr().out


# Configuration checks


def test_missing_space_name_raises(env, monkeypatch):
    monkeypatch.delenv(BUCKET_NAME_ENV)
    with pytest.raises(RuntimeError, match=BUCKET_NAME_ENV):
        S3Uploader()


@pytest.mark.parametrize(
    "endpoint",
    [
        f"https://{SPACE}.nyc3.digitaloceanspaces.com",  # Space in the host
        f"https://nyc3.digitaloceanspaces.com/{SPACE}",  # Space in the path
        "https://nyc3.digitaloceanspaces.com/anything/",  # Any path
    ],
)
def test_space_specific_endpoints_are_rejected(env, monkeypatch, endpoint):
    monkeypatch.setenv("FSF_FRONT_END_BUCKET_ENDPOINT", endpoint)
    with pytest.raises(ValueError, match="region-only"):
        S3Uploader()


@pytest.mark.parametrize(
    "endpoint",
    [REGION_ONLY_ENDPOINT, REGION_ONLY_ENDPOINT + "/", None, ""],
)
def test_region_only_or_default_endpoints_are_accepted(endpoint):
    validate_endpoint(endpoint, SPACE)  # Does not raise.
