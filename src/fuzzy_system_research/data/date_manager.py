from datetime import datetime
from functools import lru_cache
import exchange_calendars as xcals
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

# ISO 10383 market identifier code of the New York Stock Exchange.
NYSE_CALENDAR_CODE = "XNYS"


@lru_cache(maxsize=1)
def nyse_holidays() -> pd.DatetimeIndex:
    """
    Weekdays on which the NYSE is closed (full-day holidays and one-off
    closures), from the exchange_calendars NYSE calendar.

    Computed as all weekdays between the calendar's first and last session
    minus the actual trading sessions. The calendar covers roughly the past
    20 years through about one year ahead; outside that range no holidays
    are known and every weekday is treated as a trading day. Early-close
    (half) days are trading sessions and are not included. The result is
    cached because building the calendar is relatively slow.

    Returns
    -------
    pd.DatetimeIndex
        Tz-naive, midnight-normalized dates of NYSE weekday closures.
    """
    cal = xcals.get_calendar(NYSE_CALENDAR_CODE)
    weekdays = pd.bdate_range(cal.first_session, cal.last_session)
    return weekdays.difference(cal.sessions)


def nyse_sessions_ending(
    last_session: pd.Timestamp, n_sessions: int
) -> pd.DatetimeIndex:
    """
    The n_sessions consecutive NYSE trading sessions ending on last_session.

    Parameters
    ----------
    last_session : pd.Timestamp
        The last session. Any time of day is dropped.

    n_sessions : int
        Number of sessions, at least 1.

    Returns
    -------
    pd.DatetimeIndex
        Midnight-normalized dates, in ascending order.
    """
    return pd.date_range(
        end=pd.Timestamp(last_session).normalize(),
        periods=n_sessions,
        freq=CustomBusinessDay(holidays=nyse_holidays()),
    )


class DateManager:
    """
    Class to manage ranges of business days according to the New York
    Stock Exchange (NYSE) trading calendar.
    """

    def __init__(self):
        """
        Instantiates the class and sets up the calendar of business days
        to use: weekdays excluding NYSE holidays (see nyse_holidays()).
        """
        self.cbd = CustomBusinessDay(holidays=nyse_holidays())

    def get_today_date(self):
        """
        Return today's date as a YYYY-MM-DD string.
        """
        today = datetime.now()
        return today.strftime("%Y-%m-%d")

    def past_business_day(self, reference_date, business_days_past):
        """
        Calculates the business date a specified number of business days in the past,
        skipping NYSE holidays.

        Parameters
        ----------
        reference_date : pd.Timestamp
            Reference date to calculate days from.

        business_days_past : int
            The number of business days in the past.

        Returns
        -------
        pd.Timestamp
            The calculated past business date.
        """
        return reference_date - (self.cbd * business_days_past)

    def future_business_day(self, reference_date, business_days_ahead):
        """
        Calculates the business date a specified number of business days in the future,
        skipping NYSE holidays.

        Parameters
        ----------
        reference_date : pd.Timestamp
            Reference date to calculate days from.

        business_days_ahead : int
            The number of business days in the future.

        Returns
        -------
        pd.Timestamp
            The calculated future business date.
        """
        return reference_date + (self.cbd * business_days_ahead)

    def create_business_day_range(self, reference_date, num_days):
        """
        Creates a range of business days starting from a given date,
        skipping NYSE holidays.

        Parameters
        ----------
        start_date : pd.Timestamp
            The starting date

        num_days : int
            The number of business days to generate.

        Returns
        -------
        pd.DatetimeIndex
            A DatetimeIndex containing the range of business days.
        """
        return pd.date_range(start=reference_date, periods=num_days, freq=self.cbd)
