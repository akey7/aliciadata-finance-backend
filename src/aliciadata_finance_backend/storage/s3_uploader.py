import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# Name of the environment variable holding the name of the one frontend
# S3 bucket (on the Garage server; see "Deployment" in docs/SYSTEM_DESIGN.md).
# The two top-level folders inside it are named by
# S3_PORTFOLIO_OPTIMIZATION_FOLDER and S3_TIME_SERIES_FOLDER.
BUCKET_NAME_ENV = "S3_BUCKET_NAME"

# boto3's standard names for the S3 server's endpoint, region, and the backend's
# read/write/delete key. They are read here and passed to boto3 explicitly,
# and every one must be set, so a missing value never makes boto3 fall back
# to other credentials on the machine (such as ~/.aws/credentials) or to
# AWS's own endpoint.
ENDPOINT_ENV = "AWS_ENDPOINT_URL_S3"
REGION_ENV = "AWS_DEFAULT_REGION"
ACCESS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
SECRET_ACCESS_KEY_ENV = "AWS_SECRET_ACCESS_KEY"

# Error codes with which S3 servers (including Garage) answer HeadObject for
# a key that does not exist.
NOT_FOUND_CODES = ("404", "NoSuchKey", "NotFound")

# Every variable S3Uploader() needs, in the order they are reported missing.
REQUIRED_ENVS = (
    BUCKET_NAME_ENV,
    ENDPOINT_ENV,
    REGION_ENV,
    ACCESS_KEY_ID_ENV,
    SECRET_ACCESS_KEY_ENV,
)


def validate_endpoint(endpoint_url: str | None, bucket_name: str) -> None:
    """
    Raise ValueError unless endpoint_url is the base URL of the S3 API, such
    as https://<s3-host>, with no bucket name in its host and no path.

    An endpoint that already names the bucket, in its host
    (https://<bucket>.<s3-host>) or its path (https://<s3-host>/<bucket>),
    would make boto3 address the bucket twice, so objects would silently land
    under an extra "<bucket>/" prefix. Any other path is rejected too.

    Parameters
    ----------
    endpoint_url : str or None
        The configured endpoint. None (boto3's default endpoint) is allowed.

    bucket_name : str
        Name of the bucket.
    """
    if not endpoint_url:
        return
    parsed = urlparse(endpoint_url)
    host_labels = (parsed.hostname or "").split(".")
    if host_labels[0] == bucket_name or parsed.path.strip("/"):
        raise ValueError(
            f"{ENDPOINT_ENV} must be the base URL of the S3 API "
            "(e.g. https://<s3-host>), with no bucket name in its host and "
            f"no path; {BUCKET_NAME_ENV} names the bucket."
        )


class S3Uploader:
    """
    Manages uploads of optimization and modeling data to the one frontend
    S3 bucket.
    """

    def __init__(self):
        """
        Instantiate class and acquire values from environment variables
        (REQUIRED_ENVS). The .env file is loaded with override=True, so its
        values win over any of the same names already set in the shell (for
        example AWS credentials exported for other work).

        Raises
        ------
        RuntimeError
            If any variable in REQUIRED_ENVS is not set, naming every one
            that is missing.

        ValueError
            If AWS_ENDPOINT_URL_S3 is not a valid endpoint (see
            validate_endpoint()).
        """
        load_dotenv(override=True)
        values = {env: os.getenv(env) for env in REQUIRED_ENVS}
        missing = [env for env, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing environment variable(s) {', '.join(missing)}; "
                "set them in .env."
            )
        self.bucket_name = values[BUCKET_NAME_ENV]
        self.endpoint_url = values[ENDPOINT_ENV]
        self.region_name = values[REGION_ENV]
        self.aws_access_key_id = values[ACCESS_KEY_ID_ENV]
        self.aws_secret_access_key = values[SECRET_ACCESS_KEY_ENV]
        validate_endpoint(self.endpoint_url, self.bucket_name)

    def upload_file(self, local_filename: str, object_key: str) -> None:
        """
        Uploads a file to the frontend bucket. Prints a message after a
        successful upload.

        Errors are not caught: any failure propagates to the caller, so a
        failed upload stops the run instead of being reported as a success.

        Parameters
        ----------
        local_filename : str
            Local path to file.

        object_key : str
            Full object key in the bucket, starting with the top-level folder,
            e.g. "time-series/2026-10-09/all_forecasts.csv" (see
            storage.layout.object_key()).

        Raises
        ------
        FileNotFoundError
            If local_filename does not exist.

        botocore.exceptions.NoCredentialsError
            If no credentials are available.

        boto3.exceptions.S3UploadFailedError, botocore.exceptions.ClientError
            If S3 rejects the upload (e.g. a wrong bucket name or
            insufficient permissions).
        """
        self._client().upload_file(
            local_filename,
            self.bucket_name,
            object_key,
        )
        print(f"Upload of {object_key} successful!")

    def object_exists(self, object_key: str) -> bool:
        """
        Whether an object with this key exists in the frontend bucket, from a
        HeadObject request (which reads no content).

        Parameters
        ----------
        object_key : str
            Full object key in the bucket, as for upload_file().

        Returns
        -------
        bool
            True if the object exists, False if S3 answers that it does not.

        Raises
        ------
        botocore.exceptions.ClientError
            For any other error, such as insufficient permissions, so a
            failed check is never taken to mean "missing" or "present".
        """
        try:
            self._client().head_object(Bucket=self.bucket_name, Key=object_key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in NOT_FOUND_CODES:
                return False
            raise
        return True

    def _client(self):
        """An S3 client for the bucket, with this instance's settings."""
        session = boto3.session.Session()
        return session.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )
