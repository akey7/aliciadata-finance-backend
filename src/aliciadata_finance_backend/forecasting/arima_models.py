import warnings
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.tsa.api as tsa
from aliciadata_finance_backend.data.date_manager import DateManager

# Number of times prices are differenced before modeling: once, by
# ArimaModels.to_log_returns(). The models are ARIMA(p, 0, q) on log returns,
# which is ARIMA(p, PRICE_DIFFERENCING_ORDER, q) on log prices.
PRICE_DIFFERENCING_ORDER = 1


class ArimaModels:
    def __init__(self):
        """
        Initialize a new ArimaModels instance to manage ARIMA models.

        Everything here runs in the calling process. Parallelism lives one
        level up, in ArimaWalkForward, which fits many windows at once.
        """
        self.train_result = None
        self.final_model = None
        self.order_tie_broken = None

    def next_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Finds the next NYSE trading day, skipping weekends and NYSE holidays.
        The time of day of the given date is preserved.

        Parameters
        ----------
        date : pd.Timestamp
            The starting date.

        Returns
        -------
        pd.Timestamp
            The next NYSE trading day after the given date.
        """
        return DateManager().future_business_day(date, 1)

    def fit(self, original_ts, max_p=5, max_q=5):
        """
        Fit one ARIMA(p, 0, q) model per candidate order on the whole
        training window, choose the order with the lowest BIC (see
        select_order()), and forecast one step ahead with that order's model.
        The prices in original_ts are converted to log returns by
        to_log_returns(), the only differencing step, before fitting. The
        selected model is the one already fit during selection; it is not
        refit.

        Sets the instance attributes train_result (see train_arima()),
        final_model (the selected fitted model), and order_tie_broken
        (whether select_order() needed its tie-break rules; see
        order_was_tie_broken()).

        This assumes one index entry per NYSE trading session (as produced
        by asfreq(DateManager().cbd)) and predicts the next NYSE trading day
        (see next_trading_day()).

        Parameters
        ----------
        original_ts : pd.Series
            The original time series (prices) of the training window.

        max_p : int, optional
            Exclusive upper bound on p. Defaults to 5.

        max_q : int, optional
            Exclusive upper bound on q. Defaults to 5.

        Returns
        -------
        tuple[pd.Timestamp, float, tuple[int, int]]
            The date being forecast (the next NYSE trading day after the
            last index entry), the forecast price in the units of
            original_ts, and the (p, q) order of the model that made the
            forecast (see select_order()). The model is ARIMA(p, 0, q) on
            log returns, equivalent to ARIMA(p, 1, q) on log prices.

        Raises
        ------
        ValueError
            If every candidate's fit failed, so there is no model to
            forecast with.
        """
        ts = self.to_log_returns(original_ts)
        self.train_result, models = self.train_arima(ts, max_p=max_p, max_q=max_q)
        best_order = self.select_order(self.train_result)
        if models[best_order] is None:
            raise ValueError(
                f"Every candidate ARIMA fit failed on the window ending {ts.index[-1]}."
            )
        self.order_tie_broken = self.order_was_tie_broken(self.train_result)
        self.final_model = models[best_order]
        pred_date = self.next_trading_day(ts.index[-1])
        pred = self.predict(original_ts)
        return pred_date, pred, best_order

    def to_log_returns(self, original_ts: pd.Series) -> pd.Series:
        """
        Convert prices to daily log returns, log(P_t) - log(P_{t-1}). This
        is the only differencing step in the model: the ARIMA models are fit
        with d = 0 on these returns, which is ARIMA(p, 1, q) on log prices.

        The first value, which has no previous price, is dropped. The
        datetime index (and its frequency) is otherwise passed through, so a
        session missing from the prices still appears as NaN.

        Parameters
        ----------
        original_ts : pd.Series
            Prices, one entry per trading session.

        Returns
        -------
        pd.Series
            Log returns, one entry shorter than original_ts.
        """
        return np.log(original_ts).diff().iloc[1:]

    def train_arima(
        self, ts: pd.Series, max_p: int = 5, max_q: int = 5
    ) -> tuple[pd.DataFrame, dict]:
        """
        Fit ARIMA(p, 0, q) on the whole of ts for every p in range(max_p)
        and q in range(max_q), excluding (0, 0), one after another.

        Parameters
        ----------
        ts : pd.Series
            The time series to be modeled. The raw prices should be
            transformed by to_log_returns() first.

        max_p : int, optional
            Exclusive upper bound on p. Defaults to 5.

        max_q : int, optional
            Exclusive upper bound on q. Defaults to 5.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            The results, indexed by (p, q), with the columns of
            fit_order()'s result row; and the fitted models keyed by (p, q),
            None where the fit failed.
        """
        rows = []
        models = {}
        for p in range(max_p):
            for q in range(max_q):
                if p == 0 and q == 0:
                    continue
                model, row = self.fit_order(ts, p, q)
                rows.append(row)
                models[(p, q)] = model
        train_result = pd.DataFrame(rows).set_index(["p", "q"])
        return train_result, models

    def fit_order(self, ts: pd.Series, p: int, q: int):
        """
        Fit one ARIMA(p, 0, q) model on the whole of ts and record its
        information criteria.

        If the fit fails (LinAlgError or ValueError), the model is None, aic
        and bic are infinite so the order ranks last in select_order(), and
        fit_error names the exception.

        Parameters
        ----------
        ts : pd.Series
            Log returns (see to_log_returns()).

        p, q : int
            Autoregressive and moving-average orders.

        Returns
        -------
        tuple[statsmodels results or None, dict]
            The fitted model (None on failure), and a result row with keys
            p, q, aic, bic, and fit_error (None on success).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = tsa.ARIMA(endog=ts, order=(p, 0, q)).fit()
            except (LinAlgError, ValueError) as error:
                row = {
                    "p": p,
                    "q": q,
                    "aic": np.inf,
                    "bic": np.inf,
                    "fit_error": type(error).__name__,
                }
                return None, row
        row = {"p": p, "q": q, "aic": model.aic, "bic": model.bic, "fit_error": None}
        return model, row

    def select_order(self, train_result: pd.DataFrame) -> tuple[int, int]:
        """
        Choose the (p, q) order with the lowest BIC, each candidate having
        been fit once on the whole training window (see train_arima()). Pure
        function.

        Exactly equal BICs are rare (in practice, two failed fits, both
        infinite). Ties are broken, in this order, by:

        1. fewer parameters, p + q (parsimony),
        2. lower p (only so that the result never depends on row order).

        The result therefore does not depend on the row order of
        train_result. No tolerance is applied: a BIC lower by a tiny amount
        still wins.

        Parameters
        ----------
        train_result : pd.DataFrame
            Training results from train_arima(), indexed by (p, q), with at
            least a "bic" column.

        Returns
        -------
        tuple[int, int]
            The selected (p, q).
        """
        p = train_result.index.get_level_values("p").to_numpy()
        q = train_result.index.get_level_values("q").to_numpy()
        # Column names must differ from the index level names "p" and "q",
        # or pandas cannot tell which one a sort key refers to.
        candidates = pd.DataFrame(
            {
                "bic": train_result["bic"],
                "n_params": p + q,
                "ar_order": p,
            },
            index=train_result.index,
        )
        best_p, best_q = candidates.sort_values(
            ["bic", "n_params", "ar_order"], kind="stable"
        ).index[0]
        return int(best_p), int(best_q)

    def order_was_tie_broken(self, train_result: pd.DataFrame) -> bool:
        """
        Whether select_order() needed its tie-break rules for train_result,
        i.e. more than one candidate shares the lowest BIC. Pure function.

        Parameters
        ----------
        train_result : pd.DataFrame
            Training results from train_arima(), with a "bic" column.

        Returns
        -------
        bool
            True if two or more candidates tie for the lowest BIC.
        """
        bic = train_result["bic"]
        return bool((bic == bic.min()).sum() > 1)

    def predict(self, orginal_ts):
        """
        Predict the next time step beyond the training of the model.
        Return the prediction in units of the original time series.

        Parameters
        ----------
        original_ts : pd.Series
            Original time series, neither transformed nor differenced

        Returns
        -------
        np.float64
            The next value in the time series.
        """
        forecast = self.final_model.forecast(steps=1)
        pred = orginal_ts.iloc[-1] * np.exp(forecast.iloc[0])
        return pred
