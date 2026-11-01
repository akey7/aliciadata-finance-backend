"""
Tests for the output and bucket key layout in fuzzy_system_research.storage.layout.

In-memory only, except ensure_parent_dir, which writes under pytest's
tmp_path. Never touches the real input/ or output/ folders.
"""

import os

import pandas as pd
import pytest

from fuzzy_system_research.storage.layout import (
    ALL_FORECASTS_FILENAME,
    ALL_FORECASTS_METADATA_FILENAME,
    OPTIMIZATION_METADATA_FILENAME,
    OPTIMIZATION_PLOT_DATA_FILENAME,
    OUTPUT_ROOT,
    PORTFOLIO_BACKTEST_DAILY_FILENAME,
    PORTFOLIO_BACKTEST_METADATA_FILENAME,
    PORTFOLIO_OPTIMIZATION,
    TIME_SERIES,
    as_of_date_str,
    bucket_key,
    ensure_parent_dir,
    local_output_path,
    object_key,
)


# File names are part of the frontend contract.


def test_contract_file_names():
    assert OPTIMIZATION_METADATA_FILENAME == "optimization_metadata.yml"
    assert OPTIMIZATION_PLOT_DATA_FILENAME == "portfolio_optimization_plot_data.h5"
    assert ALL_FORECASTS_FILENAME == "all_forecasts.csv"
    assert ALL_FORECASTS_METADATA_FILENAME == "all_forecasts_metadata.yml"
    assert PORTFOLIO_BACKTEST_DAILY_FILENAME == "portfolio_backtest_daily.csv"
    assert PORTFOLIO_BACKTEST_METADATA_FILENAME == "portfolio_backtest_metadata.yml"


# as_of_date_str


def test_as_of_date_str_uses_latest_date_and_drops_time():
    index = pd.DatetimeIndex(
        ["2026-10-08 17:00:00", "2026-10-12 17:00:00", "2026-10-09 17:00:00"]
    )
    assert as_of_date_str(index) == "2026-10-12"


def test_as_of_date_str_rejects_empty_index():
    with pytest.raises(ValueError):
        as_of_date_str(pd.DatetimeIndex([]))


# bucket_key


def test_bucket_key():
    assert bucket_key("2026-10-12", ALL_FORECASTS_FILENAME) == (
        "2026-10-12/all_forecasts.csv"
    )


@pytest.mark.parametrize(
    "bad_date", ["2026-1-5", "20261012", "2026-02-30", "2026/10/12", "", None]
)
def test_bucket_key_rejects_bad_dates(bad_date):
    with pytest.raises(ValueError):
        bucket_key(bad_date, ALL_FORECASTS_FILENAME)


@pytest.mark.parametrize("bad_name", ["", "a/b.csv", "..", ".", "a\\b.csv"])
def test_bucket_key_rejects_bad_filenames(bad_name):
    with pytest.raises(ValueError):
        bucket_key("2026-10-12", bad_name)


# object_key


def test_object_key_is_folder_date_file():
    assert object_key("time-series", "2026-10-09", ALL_FORECASTS_FILENAME) == (
        "time-series/2026-10-09/all_forecasts.csv"
    )
    assert object_key(
        "portfolio-optimization", "2026-10-09", OPTIMIZATION_METADATA_FILENAME
    ) == ("portfolio-optimization/2026-10-09/optimization_metadata.yml")


def test_object_key_is_folder_plus_bucket_key():
    assert object_key("f", "2026-10-12", "x.csv") == "f/" + bucket_key(
        "2026-10-12", "x.csv"
    )


@pytest.mark.parametrize("bad_folder", ["", None, "a/b", "..", ".", "a\\b"])
def test_object_key_rejects_bad_folders(bad_folder):
    with pytest.raises(ValueError):
        object_key(bad_folder, "2026-10-12", ALL_FORECASTS_FILENAME)


def test_object_key_validates_date_and_filename():
    with pytest.raises(ValueError):
        object_key("f", "2026-1-5", ALL_FORECASTS_FILENAME)
    with pytest.raises(ValueError):
        object_key("f", "2026-10-12", "a/b.csv")


# local_output_path


def test_local_output_path_portfolio_optimization():
    path = local_output_path(
        PORTFOLIO_OPTIMIZATION, "2026-10-12", OPTIMIZATION_METADATA_FILENAME
    )
    assert path == os.path.join(
        "output", "portfolio_optimization", "2026-10-12", "optimization_metadata.yml"
    )


def test_local_output_path_time_series():
    path = local_output_path(TIME_SERIES, "2026-10-12", ALL_FORECASTS_FILENAME)
    assert path == os.path.join(
        "output", "time_series", "2026-10-12", "all_forecasts.csv"
    )


def test_local_output_path_custom_root(tmp_path):
    path = local_output_path(
        TIME_SERIES, "2026-10-12", ALL_FORECASTS_FILENAME, output_root=str(tmp_path)
    )
    assert path == os.path.join(
        str(tmp_path), "time_series", "2026-10-12", "all_forecasts.csv"
    )


def test_local_output_path_rejects_unknown_kind():
    with pytest.raises(ValueError):
        local_output_path("arima", "2026-10-12", ALL_FORECASTS_FILENAME)


@pytest.mark.parametrize(
    "kind, filename",
    [
        (PORTFOLIO_OPTIMIZATION, OPTIMIZATION_METADATA_FILENAME),
        (PORTFOLIO_OPTIMIZATION, OPTIMIZATION_PLOT_DATA_FILENAME),
        (TIME_SERIES, ALL_FORECASTS_FILENAME),
    ],
)
def test_local_path_mirrors_bucket_key(kind, filename):
    # The local path minus "output/<kind>/" must equal the bucket key.
    local = local_output_path(kind, "2026-10-12", filename)
    relative = os.path.relpath(local, os.path.join(OUTPUT_ROOT, kind))
    assert relative.replace(os.sep, "/") == bucket_key("2026-10-12", filename)


# ensure_parent_dir


def test_ensure_parent_dir_creates_missing_folders(tmp_path):
    path = local_output_path(
        TIME_SERIES, "2026-10-12", ALL_FORECASTS_FILENAME, output_root=str(tmp_path)
    )
    assert not os.path.exists(os.path.dirname(path))
    assert ensure_parent_dir(path) == path
    assert os.path.isdir(os.path.dirname(path))
    assert not os.path.exists(path)  # Creates folders only, never the file.


def test_ensure_parent_dir_is_idempotent(tmp_path):
    path = os.path.join(str(tmp_path), "a", "b", "file.csv")
    ensure_parent_dir(path)
    ensure_parent_dir(path)
    assert os.path.isdir(os.path.join(str(tmp_path), "a", "b"))


def test_ensure_parent_dir_bare_filename_is_noop():
    assert ensure_parent_dir("file.csv") == "file.csv"
