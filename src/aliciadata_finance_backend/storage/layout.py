"""
Where each run's results live: locally under output/ and remotely in the one
frontend Space (S3 bucket). Both layouts share one date folder per run,
named by the data as-of date (the last NYSE trading day in the run's price
data):

    output/<kind>/YYYY-MM-DD/<filename>       (local)
    <folder>/YYYY-MM-DD/<filename>            (object key in the Space)

where <kind> is "portfolio_optimization" or "time_series", and <folder> is
the top-level folder in the Space for that kind (the value of
FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER or FSF_FRONT_END_TIME_SERIES_FOLDER). The local path
minus its "output/<kind>/" part equals the object key minus its "<folder>/"
part, so the two layouts cannot drift apart. All functions here are pure
except ensure_parent_dir().
"""

import os
import re
from datetime import datetime
import pandas as pd

OUTPUT_ROOT = "output"

PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
TIME_SERIES = "time_series"
OUTPUT_KINDS = (PORTFOLIO_OPTIMIZATION, TIME_SERIES)

OPTIMIZATION_METADATA_FILENAME = "optimization_metadata.yml"
OPTIMIZATION_PLOT_DATA_FILENAME = "portfolio_optimization_plot_data.h5"
ALL_FORECASTS_FILENAME = "all_forecasts.csv"
ALL_FORECASTS_METADATA_FILENAME = "all_forecasts_metadata.yml"
PORTFOLIO_BACKTEST_DAILY_FILENAME = "portfolio_backtest_daily.csv"
PORTFOLIO_BACKTEST_METADATA_FILENAME = "portfolio_backtest_metadata.yml"

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_as_of(as_of: str) -> None:
    """
    Raise ValueError unless as_of is a real calendar date written exactly
    as YYYY-MM-DD.
    """
    if not isinstance(as_of, str) or not _DATE_PATTERN.match(as_of):
        raise ValueError(f"as_of must be a YYYY-MM-DD string, got {as_of!r}")
    datetime.strptime(as_of, "%Y-%m-%d")  # Rejects e.g. 2026-02-30


def _validate_filename(filename: str) -> None:
    """
    Raise ValueError unless filename is a single, non-empty path component.
    """
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or filename in (".", "..")
    ):
        raise ValueError(f"filename must be a plain file name, got {filename!r}")


def as_of_date_str(index: pd.DatetimeIndex) -> str:
    """
    Return the data as-of date of a run: the latest date in the index of
    its price data, formatted as YYYY-MM-DD. Any time of day is dropped.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index of the run's (wide) price data.

    Returns
    -------
    str
        The latest date in the index as YYYY-MM-DD.

    Raises
    ------
    ValueError
        If the index is empty.
    """
    if len(index) == 0:
        raise ValueError("Cannot determine an as-of date from an empty index")
    return pd.Timestamp(index.max()).strftime("%Y-%m-%d")


def bucket_key(as_of: str, filename: str) -> str:
    """
    Return the key of a run's file relative to its top-level folder in the
    Space. Use object_key() for the full object key.

    Parameters
    ----------
    as_of : str
        Data as-of date, YYYY-MM-DD.

    filename : str
        File name, e.g. "all_forecasts.csv".

    Returns
    -------
    str
        "YYYY-MM-DD/<filename>".
    """
    _validate_as_of(as_of)
    _validate_filename(filename)
    return f"{as_of}/{filename}"


def object_key(folder: str, as_of: str, filename: str) -> str:
    """
    Return the full object key of a run's file in the frontend Space, as
    specified in docs/INTERFACE_CONTRACT.md.

    Parameters
    ----------
    folder : str
        Top-level folder in the Space, i.e. the value of
        FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER or FSF_FRONT_END_TIME_SERIES_FOLDER.

    as_of : str
        Data as-of date, YYYY-MM-DD.

    filename : str
        File name, e.g. "all_forecasts.csv".

    Returns
    -------
    str
        "<folder>/YYYY-MM-DD/<filename>".
    """
    if not isinstance(folder, str) or not folder:
        raise ValueError(f"folder must be a non-empty string, got {folder!r}")
    _validate_filename(folder)  # A single path component, like a file name.
    return f"{folder}/{bucket_key(as_of, filename)}"


def local_output_path(
    kind: str, as_of: str, filename: str, output_root: str = OUTPUT_ROOT
) -> str:
    """
    Return the local path of a run's output file. Does not create anything.

    Parameters
    ----------
    kind : str
        PORTFOLIO_OPTIMIZATION or TIME_SERIES.

    as_of : str
        Data as-of date, YYYY-MM-DD.

    filename : str
        File name, e.g. "all_forecasts.csv".

    output_root : str, optional
        Root output folder, relative to the working directory. Defaults to
        "output".

    Returns
    -------
    str
        "<output_root>/<kind>/YYYY-MM-DD/<filename>".
    """
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"kind must be one of {OUTPUT_KINDS}, got {kind!r}")
    _validate_as_of(as_of)
    _validate_filename(filename)
    return os.path.join(output_root, kind, as_of, filename)


def ensure_parent_dir(path: str) -> str:
    """
    Create the parent folder of path (and any missing ancestors) if it does
    not exist yet.

    Parameters
    ----------
    path : str
        Path of a file about to be written.

    Returns
    -------
    str
        The same path, for convenient chaining.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
