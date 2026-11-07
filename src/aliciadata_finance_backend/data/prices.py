"""
Pure transformations of downloaded price data shared by the forecasting and
optimization workflows.
"""

import pandas as pd


def pivot_close_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots the adjusted close values retrieved by
    TickerDownloadManager.get_latest_tickers() to a wide format suitable for
    modeling.

    Every timestamp is set to 17:00:00 on its date, a fixed label with no
    meaning beyond the date.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long DataFrame of ticker data with "datetime", "ticker", and "close"
        either as columns or as the index plus columns.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with adjusted close prices of tickers in the columns
        (sorted by ticker) and a sorted pd.DatetimeIndex suitable for
        indexing into ranges of dates.
    """
    wide_df = long_df.reset_index().pivot(
        index="datetime", columns="ticker", values="close"
    )
    wide_df.index = pd.DatetimeIndex(wide_df.index)
    wide_df.index = pd.DatetimeIndex(
        [dt.replace(hour=17, minute=0, second=0) for dt in wide_df.index]
    )
    wide_df.sort_index(inplace=True)
    return wide_df


def last_n_sessions(wide_df: pd.DataFrame, n_sessions: int) -> pd.DataFrame:
    """
    Return the most recent n_sessions rows (trading sessions) of a wide price
    DataFrame, in date order. If there are fewer rows, all rows are returned.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide price data with a DatetimeIndex.

    n_sessions : int
        Number of most recent trading sessions to keep. Must be positive.

    Returns
    -------
    pd.DataFrame
        A copy of the last n_sessions rows, sorted by date.
    """
    if n_sessions <= 0:
        raise ValueError(f"n_sessions must be positive, got {n_sessions}")
    return wide_df.sort_index().iloc[-n_sessions:].copy()
