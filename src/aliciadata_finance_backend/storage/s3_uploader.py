import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import boto3
from botocore.client import Config

# Name of the environment variable holding the name of the one frontend
# Space (S3 bucket). The two top-level folders inside it are named by
# FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER and FSF_FRONT_END_TIME_SERIES_FOLDER.
BUCKET_NAME_ENV = "FSF_FRONT_END_BUCKET_NAME"


def validate_endpoint(endpoint_url: str | None, bucket_name: str) -> None:
    """
    Raise ValueError unless endpoint_url is a region-only Spaces endpoint
    such as https://nyc3.digitaloceanspaces.com.

    An endpoint that already names the Space, in its host
    (https://<space>.<region>...) or its path (https://<region>.../<space>),
    would make boto3 address the Space twice, so objects would silently land
    under an extra "<space>/" prefix.

    Parameters
    ----------
    endpoint_url : str or None
        The configured endpoint. None (boto3's default endpoint) is allowed.

    bucket_name : str
        Name of the Space.
    """
    if not endpoint_url:
        return
    parsed = urlparse(endpoint_url)
    host_labels = (parsed.hostname or "").split(".")
    if host_labels[0] == bucket_name or parsed.path.strip("/"):
        raise ValueError(
            "FSF_FRONT_END_BUCKET_ENDPOINT must be the region-only endpoint "
            "(e.g. https://nyc3.digitaloceanspaces.com) without the Space name; "
            f"{BUCKET_NAME_ENV} names the Space."
        )


class S3Uploader:
    """
    Manages uploads of optimization and modeling data to the one frontend
    Space (S3 bucket).
    """

    def __init__(self):
        """
        Instantiate class and acquire values from environment variables.

        Raises
        ------
        RuntimeError
            If FSF_FRONT_END_BUCKET_NAME is not set.

        ValueError
            If FSF_FRONT_END_BUCKET_ENDPOINT is not a region-only endpoint
            (see validate_endpoint()).
        """
        load_dotenv()
        self.endpoint_url = os.getenv("FSF_FRONT_END_BUCKET_ENDPOINT")
        self.aws_access_key_id = os.getenv("FSF_FRONT_END_BUCKET_KEY_ID")
        self.aws_secret_access_key = os.getenv("FSF_FRONT_END_BUCKET_RWDELETE")
        self.region_name = os.getenv("FSF_FRONT_END_BUCKET_REGION")
        self.bucket_name = os.getenv(BUCKET_NAME_ENV)
        if not self.bucket_name:
            raise RuntimeError(f"Environment variable {BUCKET_NAME_ENV} is not set.")
        validate_endpoint(self.endpoint_url, self.bucket_name)

    def upload_file(self, local_filename: str, object_key: str) -> None:
        """
        Uploads a file to the frontend Space. Prints a message after a
        successful upload.

        Errors are not caught: any failure propagates to the caller, so a
        failed upload stops the run instead of being reported as a success.

        Parameters
        ----------
        local_filename : str
            Local path to file.

        object_key : str
            Full object key in the Space, starting with the top-level folder,
            e.g. "time-series/2026-10-09/all_forecasts.csv" (see
            storage.layout.object_key()).

        Raises
        ------
        FileNotFoundError
            If local_filename does not exist.

        botocore.exceptions.NoCredentialsError
            If no credentials are available.

        boto3.exceptions.S3UploadFailedError, botocore.exceptions.ClientError
            If S3 rejects the upload (e.g. a wrong Space name or
            insufficient permissions).
        """
        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        client.upload_file(
            local_filename,
            self.bucket_name,
            object_key,
        )
        print(f"Upload of {object_key} successful!")
