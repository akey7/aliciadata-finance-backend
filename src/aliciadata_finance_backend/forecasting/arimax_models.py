"""
ARIMAX: one-step-ahead forecasts of a ticker's daily log return from its own
past and from lagged exogenous variables (such as the S&P 500 index's daily
log return).

The model is statsmodels' ARIMA with exog, which is a regression with
ARMA(p, q) errors, not an ARMA model with the regressors inside the AR
recursion:

    r_t = c + sum_k beta_k * z_{k,t} + u_t,   u_t ~ ARMA(p, q)

where r_t is the ticker's daily log return on session t and z_{k,t} is
exogenous variable k, transformed (for example to a daily log return) and
lagged by its configured number of sessions. With a lag of at least 1, the
regressor of the session being forecast is already known at the forecast
origin, so the forecast uses no information from the forecast day itself.

As for ARIMA (arima_models.py), prices are differenced once, to log returns,
so ARIMAX(p, 0, q) on log returns is ARIMAX(p, 1, q) on log prices. Unlike
ARIMA, the candidate orders include (0, 0): a regression on the exogenous
variables with white-noise errors.

Everything here is pure computation on in-memory data; the walk-forward and
the worker pool are in ticker_predict.py.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
import statsmodels.tsa.api as tsa
from aliciadata_finance_backend.config import ExogenousVariable
from aliciadata_finance_backend.forecasting.arima_models import ArimaModels

# How each transform in config.EXOGENOUS_TRANSFORMS turns an exogenous
# variable's levels into the series that is lagged and regressed on. Each
# transform's first value is missing (it needs the previous level), which
# build_arimax_window() relies on when it drops the first sessions.
EXOGENOUS_TRANSFORM_FUNCTIONS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "log_return": lambda levels: np.log(levels).diff(),
}


def transform_exogenous(levels: pd.Series, transform: str) -> pd.Series:
    """
    Apply one of EXOGENOUS_TRANSFORM_FUNCTIONS to an exogenous variable's
    levels. Pure function.

    Parameters
    ----------
    levels : pd.Series
        The variable's values (for an index, its daily closes), one entry
        per session. Missing values stay missing, and make the transformed
        values that depend on them missing too.

    transform : str
        A key of EXOGENOUS_TRANSFORM_FUNCTIONS, such as "log_return"
        (log(x_t) - log(x_{t-1})).

    Returns
    -------
    pd.Series
        The transformed series on the same index; its first value is
        missing.

    Raises
    ------
    ValueError
        If transform is not implemented.
    """
    try:
        function = EXOGENOUS_TRANSFORM_FUNCTIONS[transform]
    except KeyError:
        raise ValueError(
            f"Unsupported exogenous transform {transform!r}; implemented: "
            f"{sorted(EXOGENOUS_TRANSFORM_FUNCTIONS)}."
        ) from None
    return function(levels)


@dataclass(frozen=True)
class ArimaxWindow:
    """
    Everything one ARIMAX fit needs for one ticker and one training window.
    Built by build_arimax_window(); picklable, so worker processes can fit
    it.

    Attributes
    ----------
    prices : pd.Series
        The ticker's closes in the training window, one entry per NYSE
        session (as ArimaWalkForward builds them), ending on a present
        close.

    regressors : pd.DataFrame
        One column per exogenous variable, one row per log return the model
        is fit on: the window's log returns without the first max(lag), so
        every row's lagged regressor comes from a session inside the window.
        Row t holds each variable's transformed value lag sessions before t.
        Values can be missing where the variable's level is.

    forecast_regressors : pd.Series
        Each variable's regressor for the session being forecast (its
        transformed value lag sessions before it), all present.
    """

    prices: pd.Series
    regressors: pd.DataFrame
    forecast_regressors: pd.Series


def build_arimax_window(
    prices: pd.Series,
    exogenous_levels: pd.DataFrame,
    variables: tuple[ExogenousVariable, ...],
) -> ArimaxWindow | None:
    """
    Line the exogenous variables up with one ticker's training window and
    lag them. Pure function.

    Each variable's levels are taken on the window's own sessions (sessions
    missing from exogenous_levels become missing values), transformed
    (transform_exogenous()), and shifted by the variable's lag in sessions.
    The first 1 + max(lag) sessions of the window have no regressor built
    only from data inside the window, so the model is fit on the window's
    last len(prices) - 1 - max(lag) log returns: with lag 1, one return
    fewer than the ARIMA model on the same window. Every candidate order is
    fit on the same rows, so their BICs are comparable.

    Parameters
    ----------
    prices : pd.Series
        The ticker's closes in the training window, one entry per NYSE
        session, ending on a present close.

    exogenous_levels : pd.DataFrame
        The exogenous variables' levels, one column per symbol, indexed like
        the prices (the same fixed time-of-day label). Only dates up to the
        window's last session may be used, and only those are read.

    variables : tuple[ExogenousVariable, ...]
        The variables to regress on, in column order. At least one.

    Returns
    -------
    ArimaxWindow or None
        The window, or None if a variable's regressor for the session being
        forecast is missing (its level is missing on the window's last
        session, or, for a log return, on the session before it), so there
        is no ARIMAX forecast for that row.

    Raises
    ------
    ValueError
        If variables is empty, a variable's lag is below 1 (the regressor
        would come from the forecast day itself), or the window is too short
        to leave a row to fit on.
    """
    if not variables:
        raise ValueError("ARIMAX needs at least one exogenous variable.")
    if min(v.lag for v in variables) < 1:
        raise ValueError("Every exogenous lag must be at least 1 session.")
    max_lag = max(v.lag for v in variables)
    n_skip = 1 + max_lag
    if len(prices) <= n_skip:
        raise ValueError(
            f"A {len(prices)}-session window leaves no log return to fit on "
            f"with a lag of {max_lag}."
        )
    regressors = {}
    forecast = {}
    for variable in variables:
        if variable.symbol in exogenous_levels.columns:
            levels = exogenous_levels[variable.symbol].reindex(prices.index)
        else:
            levels = pd.Series(np.nan, index=prices.index)
        transformed = transform_exogenous(levels.astype(float), variable.transform)
        regressors[variable.symbol] = transformed.shift(variable.lag)
        # The session after the window's last one, lag sessions back.
        forecast[variable.symbol] = transformed.iloc[-variable.lag]
    forecast_regressors = pd.Series(forecast, dtype=float)
    if forecast_regressors.isna().any():
        return None
    regressor_df = pd.DataFrame(regressors, index=prices.index).iloc[n_skip:]
    return ArimaxWindow(
        prices=prices,
        regressors=regressor_df,
        forecast_regressors=forecast_regressors,
    )


class ArimaxModels(ArimaModels):
    """
    Fit ARIMAX(p, 0, q) on log returns for every candidate order, choose the
    lowest BIC with ArimaModels.select_order()'s tie-break rules, and
    forecast one step ahead. Reuses ArimaModels for the log returns, the
    order selection, and the forecast date.
    """

    def fit_arimax(
        self, window: ArimaxWindow, max_p: int = 2, max_q: int = 2
    ) -> tuple[pd.Timestamp, float, tuple[int, int]]:
        """
        Fit every (p, q) with p in range(max_p) and q in range(max_q),
        including (0, 0), on the window's regression rows, choose the order
        with the lowest BIC (select_order()), and forecast the next session's
        close with that order's model, which is not refit.

        A row whose regressor is missing is treated as a missing observation:
        its log return is set to missing (the state space model skips it)
        and its regressor to 0, which then has no effect. The number of rows
        stays the same for every candidate. Sets train_result, final_model,
        and order_tie_broken, as ArimaModels.fit() does.

        Parameters
        ----------
        window : ArimaxWindow
            From build_arimax_window().

        max_p, max_q : int, optional
            Exclusive upper bounds on p and q. Default to 2 (0 or 1).

        Returns
        -------
        tuple[pd.Timestamp, float, tuple[int, int]]
            The date being forecast (the next NYSE session after the
            window), the forecast close in the units of window.prices, and
            the selected (p, q).

        Raises
        ------
        ValueError
            If every candidate's fit failed.
        """
        log_returns = self.to_log_returns(window.prices)
        n_skip = len(log_returns) - len(window.regressors)
        endog = log_returns.iloc[n_skip:]
        if not endog.index.equals(window.regressors.index):
            raise ValueError("The regressors do not line up with the window's returns.")
        missing = window.regressors.isna().any(axis=1)
        endog = endog.mask(missing)
        exog = window.regressors.copy()
        exog.loc[missing, :] = 0.0
        rows = []
        models = {}
        for p in range(max_p):
            for q in range(max_q):
                model, row = self.fit_arimax_order(endog, exog, p, q)
                rows.append(row)
                models[(p, q)] = model
        self.train_result = pd.DataFrame(rows).set_index(["p", "q"])
        best_order = self.select_order(self.train_result)
        if models[best_order] is None:
            raise ValueError(
                f"Every candidate ARIMAX fit failed on the window ending "
                f"{window.prices.index[-1]}."
            )
        self.order_tie_broken = self.order_was_tie_broken(self.train_result)
        self.final_model = models[best_order]
        next_exog = window.forecast_regressors[exog.columns].to_numpy().reshape(1, -1)
        forecast = self.final_model.forecast(steps=1, exog=next_exog)
        pred = float(window.prices.iloc[-1] * np.exp(forecast.iloc[0]))
        pred_date = self.next_trading_day(window.prices.index[-1])
        return pred_date, pred, best_order

    def fit_arimax_order(self, endog: pd.Series, exog: pd.DataFrame, p: int, q: int):
        """
        Fit one ARIMAX(p, 0, q) (a constant, the regressors, and ARMA(p, q)
        errors) and record its information criteria. A failed fit
        (LinAlgError or ValueError) gives None and infinite aic and bic, as
        in ArimaModels.fit_order().

        Parameters
        ----------
        endog : pd.Series
            Log returns, missing where an observation is skipped.

        exog : pd.DataFrame
            Regressors on the same index, with no missing values.

        p, q : int
            Autoregressive and moving-average orders of the errors.

        Returns
        -------
        tuple[statsmodels results or None, dict]
            The fitted model (None on failure), and a result row with keys
            p, q, aic, bic, and fit_error (None on success).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = tsa.ARIMA(endog=endog, exog=exog, order=(p, 0, q)).fit()
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
