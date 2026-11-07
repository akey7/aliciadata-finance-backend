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
    ACCESS_KEY_ID_ENV,
    BUCKET_NAME_ENV,
    ENDPOINT_ENV,
    REGION_ENV,
    REQUIRED_ENVS,
    SECRET_ACCESS_KEY_ENV,
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
    """
    Set every required variable, and replace load_dotenv with a stub that
    records its keyword arguments instead of reading .env.
    """
    dotenv_calls: list[dict] = []
    monkeypatch.setattr(
        s3_module, "load_dotenv", lambda **kwargs: dotenv_calls.append(kwargs)
    )
    monkeypatch.setenv(BUCKET_NAME_ENV, SPACE)
    monkeypatch.setenv(ENDPOINT_ENV, REGION_ONLY_ENDPOINT)
    monkeypatch.setenv(REGION_ENV, "nyc3")
    monkeypatch.setenv(ACCESS_KEY_ID_ENV, "test-key-id")
    monkeypatch.setenv(SECRET_ACCESS_KEY_ENV, "test-secret")
    return dotenv_calls


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
    assert fake_client["kwargs"]["region_name"] == "nyc3"
    assert fake_client["kwargs"]["aws_access_key_id"] == "test-key-id"
    assert fake_client["kwargs"]["aws_secret_access_key"] == "test-secret"
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


@pytest.mark.parametrize("missing", REQUIRED_ENVS)
def test_each_missing_variable_raises(env, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        S3Uploader()


def test_empty_variable_counts_as_missing(env, monkeypatch):
    monkeypatch.setenv(SECRET_ACCESS_KEY_ENV, "")
    with pytest.raises(RuntimeError, match=SECRET_ACCESS_KEY_ENV):
        S3Uploader()


def test_every_missing_variable_is_named(env, monkeypatch):
    monkeypatch.delenv(ACCESS_KEY_ID_ENV)
    monkeypatch.delenv(SECRET_ACCESS_KEY_ENV)
    with pytest.raises(RuntimeError) as excinfo:
        S3Uploader()
    assert ACCESS_KEY_ID_ENV in str(excinfo.value)
    assert SECRET_ACCESS_KEY_ENV in str(excinfo.value)


def test_dotenv_values_override_the_shell(env):
    S3Uploader()
    assert env == [{"override": True}]


@pytest.mark.parametrize(
    "endpoint",
    [
        f"https://{SPACE}.nyc3.digitaloceanspaces.com",  # Space in the host
        f"https://nyc3.digitaloceanspaces.com/{SPACE}",  # Space in the path
        "https://nyc3.digitaloceanspaces.com/anything/",  # Any path
    ],
)
def test_space_specific_endpoints_are_rejected(env, monkeypatch, endpoint):
    monkeypatch.setenv(ENDPOINT_ENV, endpoint)
    with pytest.raises(ValueError, match="region-only"):
        S3Uploader()


@pytest.mark.parametrize(
    "endpoint",
    [REGION_ONLY_ENDPOINT, REGION_ONLY_ENDPOINT + "/", None, ""],
)
def test_region_only_or_default_endpoints_are_accepted(endpoint):
    validate_endpoint(endpoint, SPACE)  # Does not raise.
