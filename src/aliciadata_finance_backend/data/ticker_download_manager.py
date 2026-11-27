import os
import time
import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient
from aliciadata_finance_backend.data.date_manager import DateManager
from aliciadata_finance_backend.storage.layout import ensure_parent_dir


class TickerDownloadManager:
    """
    Instances of this class manage downloading ticker data from the Massive
    API over a given range of trading days.
    """

    def __init__(self, download_folder_name=".", tickers=None, dm=None):
        """
        Instantiates the tickers this instance will download and
        where the data will be saved.

        Parameters
        ----------
        download_folder_name : str, optional
            The path to the download folder. Defaults to working directory.

        tickers : list[str], optional
            The list of tickers to download. If left as none, will download
            the tickers given below.

        dm : DateManager, optional
            Supplies "today", which names the cache file and ends the
            download range. Defaults to a DateManager for the current date;
            a backfill passes one with a fixed date (see DateManager).
        """
        load_dotenv(override=True)
        massive_api_key = os.getenv("MASSIVE_API_KEY")
        self.massive_client = RESTClient(massive_api_key)
        self.dm = dm if dm is not None else DateManager()
        self.download_folder_name = download_folder_name
        if not tickers:
            self.tickers = [
                "XBI",
                "QQQ",
                "XLE",
                "VOO",
                "NVDA",
                "VXUS",
                "GLD",
                "IBIT",
                "FXE",
                "ETHA",
            ]
        else:
            self.tickers = tickers

    def download_tickers(self, tickers, date_from, date_to, delay=0):
        """
        Gets ticker data from Massive.com between the given dates, inclusive
        of the ending date.

        Parameters
        ----------
        tickers : List[str]
            List of valid ticker symbols (like AAPL)

        date_from : str
            String representation of the start date in YYYY-MM-DD format.

        date_to : str
            String representation of the end date in YYYY-MM-DD format.

        delay: int, optional
            Seconds to wait between successive API calls for the tickers.
            If not specifed defaults to no delay.

        Returns
        -------
        pd.DataFrame
            DataFrame of ticker data in long format.
        """
        rows = []
        for ticker in tickers:
            for a in self.massive_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=date_from,
                to=date_to,
                adjusted="true",
            ):
                rows.append(
                    {
                        "timestamp": a.timestamp,
                        "ticker": ticker,
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "vwap": a.vwap,
                        "transactions": a.transactions,
                    }
                )
            if delay > 0:
                print(f"{ticker}. Acquired {len(rows)} so far. Sleeping {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"{ticker}. Acquired {len(rows)} so far.")
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize(None)
        df["datetime"] = df["datetime"].apply(
            lambda x: pd.Timestamp(x).replace(hour=23, minute=59, second=59)
        )
        return df.set_index("datetime").drop(columns="timestamp").sort_index()

    def get_latest_tickers(
        self, days_in_past: int = 40, use_cache: bool = True
    ) -> tuple[pd.DataFrame, str | pd.Timestamp, str | pd.Timestamp]:
        """
        Retrieve the latest tickers, ending on the business day before today.

        Downloads are cached in the download folder in a file named
        "Tickers YYYY-MM-DD.csv" using today's date. A cache file from an
        earlier day is never reused, because its data would end before the
        most recent trading day. The download folder (and any missing
        parent folders) is created when the cache file is written, if it does
        not exist yet.

        Parameters
        ----------
        days_in_past : int
            The number of days in past to retrieve. Defaults to 40
            (2 business months.)

        use_cache : bool
            If True and today's cache file already exists, does not call the
            API and reads that file instead. If today's cache file does not
            exist, downloads new data (and writes today's cache file). If
            False, always downloads new data and overwrites today's cache file.

        Returns
        -------
        Tuple[pd.DataFrame, str | pd.Timestamp, str | pd.Timestamp]
            Returns the latest dataframe, start date, and end date. The dates
            are strings when read from the cache and pd.Timestamps when
            downloaded.
        """
        long_df_filename = os.path.join(
            self.download_folder_name, f"Tickers {self.dm.get_today_date()}.csv"
        )
        return self._latest(self.tickers, long_df_filename, days_in_past, use_cache)

    def get_latest_exogenous(
        self,
        symbols: list[str],
        folder_name: str,
        days_in_past: int = 40,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, str | pd.Timestamp, str | pd.Timestamp]:
        """
        Retrieve the latest daily values of the exogenous variables (such as
        the S&P 500 index, I:SPX), over the same range as get_latest_tickers()
        for the same days_in_past, ending on the business day before today.

        They are cached apart from the tickers being modeled, in folder_name
        (normally input/exogenous/), in a file named
        "Exogenous YYYY-MM-DD.csv" using today's date (see DateManager's
        today), in the same long format as the tickers' cache: one row per
        symbol and session, with the symbol in the "ticker" column and its
        value in "close". Each symbol is one call to Massive.com's daily
        aggregates, the same endpoint as the tickers, which returns the whole
        range at once; indices are requested by their symbol (e.g. I:SPX).
        The download is printed per symbol, like the tickers'.

        Parameters
        ----------
        symbols : list[str]
            The exogenous variables' symbols (ModelingConfig.exogenous_symbols).

        folder_name : str
            Folder of the cache file. Created when the file is written.

        days_in_past : int
            The number of business days in the past to retrieve.

        use_cache : bool
            As for get_latest_tickers(): read today's cache file if it exists,
            otherwise download and write it; if False, always download.

        Returns
        -------
        Tuple[pd.DataFrame, str | pd.Timestamp, str | pd.Timestamp]
            The values in long format, and the start and end dates, as
            returned by get_latest_tickers().
        """
        filename = os.path.join(
            folder_name, f"Exogenous {self.dm.get_today_date()}.csv"
        )
        return self._latest(list(symbols), filename, days_in_past, use_cache)

    def _latest(
        self, symbols: list[str], filename: str, days_in_past: int, use_cache: bool
    ) -> tuple[pd.DataFrame, str | pd.Timestamp, str | pd.Timestamp]:
        """
        Read today's cache file of these symbols, or download them over the
        days_in_past business days ending before today and write it (see
        get_latest_tickers() for the rules).
        """
        if use_cache and os.path.exists(filename):
            df = pd.read_csv(filename)
            start_date = df["datetime"].min()
            end_date = df["datetime"].max()
            return df, start_date, end_date
        date_from = self.dm.past_business_day(
            pd.Timestamp(self.dm.get_today_date()), days_in_past
        )
        date_to = self.dm.past_business_day(
            pd.Timestamp(self.dm.get_today_date()), 1
        ).replace(hour=23, minute=59, second=59)
        long_df = self.download_tickers(symbols, date_from=date_from, date_to=date_to)
        long_df.to_csv(ensure_parent_dir(filename), index=True)
        return long_df, date_from, date_to
