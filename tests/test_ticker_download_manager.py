"""
Unit tests for the download cache in
aliciadata_finance_backend.data.ticker_download_manager.

In-memory only. The manager is built without its constructor (so no API
client is created), the Massive.com download is replaced by a fake, and files
are written only under pytest's tmp_path, so the real input/ folder and the
network are never touched.
"""

import os

import pandas as pd

from conftest import make_long_prices
from aliciadata_finance_backend.data.ticker_download_manager import (
    TickerDownloadManager,
)

TODAY = "2026-10-14"


class FakeDateManager:
    def get_today_date(self) -> str:
        return TODAY

    def past_business_day(self, date: pd.Timestamp, n: int) -> pd.Timestamp:
        return date - pd.Timedelta(days=n)


def make_manager(download_folder_name: str) -> tuple[TickerDownloadManager, dict]:
    """A manager with a fake date manager and download; calls counts downloads."""
    calls = {"count": 0}
    tdm = TickerDownloadManager.__new__(TickerDownloadManager)
    tdm.dm = FakeDateManager()
    tdm.download_folder_name = download_folder_name
    tdm.tickers = ["XBI", "QQQ"]

    def fake_download_tickers(tickers, date_from, date_to, delay=10):
        calls["count"] += 1
        return make_long_prices(tickers=tickers, n_sessions=5).set_index("datetime")

    tdm.download_tickers = fake_download_tickers
    return tdm, calls


def test_get_latest_tickers_creates_missing_download_folder(tmp_path):
    # A fresh checkout has neither input/ nor input/annual/.
    folder = str(tmp_path / "input" / "annual")
    tdm, calls = make_manager(folder)
    long_df, _, _ = tdm.get_latest_tickers(days_in_past=5)
    cache_path = os.path.join(folder, f"Tickers {TODAY}.csv")
    assert calls["count"] == 1
    assert os.path.isfile(cache_path)
    assert len(pd.read_csv(cache_path)) == len(long_df)


def test_get_latest_tickers_reads_cache_written_into_new_folder(tmp_path):
    folder = str(tmp_path / "input" / "annual")
    tdm, calls = make_manager(folder)
    tdm.get_latest_tickers(days_in_past=5)
    cached_df, _, _ = tdm.get_latest_tickers(days_in_past=5)
    assert calls["count"] == 1
    assert set(cached_df["ticker"]) == {"XBI", "QQQ"}
