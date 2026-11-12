"""
Portfolio optimizers and the risk-free rate.

Every optimizer takes a problem.PortfolioProblem and uses its bounds(),
constraints(), daily_mean_return(), and daily_variance(), so the max Sharpe
ratio, minimum variance, maximum return, and efficient-frontier solutions
all share one constraint set and one formula for return and volatility. The
optimizers return weights only; the statistics that are uploaded come from
PortfolioProblem.portfolio_stats().

The numerical optimizations use SciPy's SLSQP, start from equal weights, and
work in daily units (percent per day, daily risk-free rate). The maximum
return portfolio has an exact solution, so it needs no solver.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pandas_datareader as web
from aliciadata_finance_backend.optimization.problem import (
    TRADING_DAYS_PER_YEAR,
    PortfolioProblem,
)
from aliciadata_finance_backend.storage.layout import ensure_parent_dir

# Calendar days of DTB3 history downloaded before the earliest optimization
# date. FRED leaves DTB3 empty on bond-market holidays, so the last rate
# before a date can be several days older than it; two weeks always contains
# one.
RISK_FREE_HISTORY_MARGIN_DAYS = 14


class OptimizationError(RuntimeError):
    """A portfolio optimization did not converge, so its result is unusable."""


def _require_success(result, name: str) -> None:
    """
    Raise OptimizationError unless the scipy.optimize.minimize() result
    reports success, naming the optimization and SciPy's status and message.
    """
    if not result.success:
        raise OptimizationError(
            f"The {name} optimization did not converge "
            f"(status {result.status}: {result.message})."
        )


def _minimize(problem: PortfolioProblem, objective, extra_constraints=()):
    """
    Minimize objective(weights) with SLSQP under the problem's bounds and
    constraints (plus extra_constraints), starting from equal weights.
    Returns SciPy's OptimizeResult.
    """
    return minimize(
        fun=objective,
        x0=problem.equal_weights(),
        method="SLSQP",
        bounds=problem.bounds(),
        constraints=[*problem.constraints(), *extra_constraints],
    )


def min_variance_weights(problem: PortfolioProblem) -> np.ndarray:
    """
    Weights of the minimum variance portfolio under the problem's
    constraints.

    Parameters
    ----------
    problem : PortfolioProblem
        The assets, estimates, and constraints.

    Returns
    -------
    np.ndarray
        Weights as fractions, shape (n,), in the order of problem.tickers.

    Raises
    ------
    OptimizationError
        If the optimizer does not converge, so no unconverged weights are
        returned.
    """
    result = _minimize(problem, problem.daily_variance)
    _require_success(result, "minimum variance")
    return result.x


def max_sharpe_weights(problem: PortfolioProblem) -> np.ndarray:
    """
    Weights of the maximum Sharpe ratio (tangency) portfolio under the
    problem's constraints. The objective is the daily Sharpe ratio, (w·μ −
    daily risk-free rate) / sqrt(wᵀΣw), which has the same maximizer as the
    annualized one.

    Parameters
    ----------
    problem : PortfolioProblem
        The assets, estimates, and constraints.

    Returns
    -------
    np.ndarray
        Weights as fractions, shape (n,), in the order of problem.tickers.

    Raises
    ------
    OptimizationError
        If the optimizer does not converge, so no unconverged weights are
        returned.
    """

    def negative_sharpe_ratio(weights: np.ndarray) -> float:
        excess = problem.daily_mean_return(weights) - problem.daily_risk_free_rate
        return -excess / np.sqrt(problem.daily_variance(weights))

    result = _minimize(problem, negative_sharpe_ratio)
    _require_success(result, "max Sharpe ratio")
    return result.x


def max_return_weights(problem: PortfolioProblem) -> np.ndarray:
    """
    Weights of the highest-return portfolio under the problem's
    constraints: the top end of the efficient frontier. With the current
    constraints it fills the highest-return assets up to the cap, in order.

    This is PortfolioProblem.max_return_weights_closed_form(), not a solver
    result. The problem is linear, so its solution is exact, while SLSQP can
    stop short of it when assets have nearly equal mean returns: the objective
    is then too flat for its convergence tolerance.

    Parameters
    ----------
    problem : PortfolioProblem
        The assets, estimates, and constraints.

    Returns
    -------
    np.ndarray
        Weights as fractions, shape (n,), in the order of problem.tickers.
    """
    return problem.max_return_weights_closed_form()


def efficient_frontier(
    problem: PortfolioProblem,
    min_daily_return: float,
    max_daily_return: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Points of the efficient frontier: for each of n_points target daily
    returns evenly spaced from min_daily_return to max_daily_return, the
    minimum variance weights whose expected return equals the target,
    under the problem's constraints. A target the optimizer does not reach
    (infeasible, or not converged) gets a row of NaN weights and is printed;
    it is not an error (see docs/INTERFACE_CONTRACT.md).

    Parameters
    ----------
    problem : PortfolioProblem
        The assets, estimates, and constraints.

    min_daily_return, max_daily_return : float
        First and last target, daily return in percent. The pipeline uses
        the minimum variance portfolio's return and the maximum feasible
        return (max_return_weights()).

    n_points : int
        Number of targets.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The target daily returns, shape (n_points,), and the weights,
        shape (n_points, n), with NaN rows for unreached targets.
    """
    print(f"Efficient frontier targets: {min_daily_return:.4f} to {max_daily_return:.4f}")
    targets = np.linspace(min_daily_return, max_daily_return, n_points)
    weights = np.full((n_points, problem.n_assets), np.nan)
    for i, target in enumerate(targets):
        target_constraint = {
            "type": "eq",
            "fun": lambda w, target=target: problem.daily_mean_return(w) - target,
        }
        result = _minimize(problem, problem.daily_variance, [target_constraint])
        if result.success:
            weights[i] = result.x
        else:
            print(f"Unreached target return: {target:.4f}")
    return targets, weights


def latest_risk_free_rate(tb3m_df) -> dict:
    """
    Pick the most recent valid 3-month T-bill rate from FRED DTB3 data and
    convert it to the daily rate used by the optimizer. Pure function.

    FRED leaves DTB3 empty on bond-market holidays (e.g. Columbus Day), which
    pandas reads as NaN; those rows are skipped so a holiday never turns the
    risk-free rate (and with it every Sharpe ratio) into NaN.

    Parameters
    ----------
    tb3m_df : pd.DataFrame
        FRED DTB3 data: a DatetimeIndex and a "DTB3" column holding the
        annualized rate in percent. Any row order.

    Returns
    -------
    dict[str, float | str]
        risk_free_rate (annualized, percent), daily_risk_free_rate
        (risk_free_rate / TRADING_DAYS_PER_YEAR, percent per day), and
        risk_free_rate_date (the observation date as a timestamp string).

    Raises
    ------
    ValueError
        If there is no non-missing DTB3 value.
    """
    valid = tb3m_df.dropna(subset=["DTB3"]).sort_index(ascending=False)
    if valid.empty:
        raise ValueError("No non-missing DTB3 (3-month T-bill) rate in the data.")
    risk_free_rate = float(valid.iloc[0]["DTB3"])
    return {
        "risk_free_rate": risk_free_rate,
        "daily_risk_free_rate": risk_free_rate / TRADING_DAYS_PER_YEAR,
        "risk_free_rate_date": str(valid.index[0]),
    }


def risk_free_rate_before(tb3m_df, cutoff) -> dict:
    """
    The most recent valid 3-month T-bill rate dated strictly before cutoff,
    in the form of latest_risk_free_rate(). A rate observed on the cutoff
    itself is excluded: FRED publishes DTB3 with a lag, so it may not have
    been available at that day's close. Every optimization uses this rule,
    live and backtest alike, so a result does not depend on the day the
    backend ran. Pure function.

    Parameters
    ----------
    tb3m_df : pd.DataFrame
        FRED DTB3 data, as for latest_risk_free_rate().

    cutoff : pd.Timestamp
        The optimization date. Any time of day is dropped.

    Returns
    -------
    dict[str, float | str]
        As returned by latest_risk_free_rate().

    Raises
    ------
    ValueError
        If there is no non-missing DTB3 value before cutoff.
    """
    return latest_risk_free_rate(tb3m_df[tb3m_df.index.normalize() < cutoff.normalize()])


def has_rate_before(tb3m_df: pd.DataFrame, first_date: pd.Timestamp) -> bool:
    """
    Whether tb3m_df has a non-missing DTB3 rate dated strictly before
    first_date, so that risk_free_rate_before(tb3m_df, first_date) succeeds.
    Pure function.
    """
    valid = tb3m_df.dropna(subset=["DTB3"])
    return bool((valid.index.normalize() < first_date.normalize()).any())


def get_risk_free_rate_history(dm, first_date: pd.Timestamp) -> pd.DataFrame:
    """
    Finds the 3-month U.S. treasury bill rates (FRED DTB3) from
    RISK_FREE_HISTORY_MARGIN_DAYS calendar days before first_date until
    today, from which every optimization takes the rate available at its
    date (see risk_free_rate_before()). first_date is the earliest
    optimization date that needs a rate, normally the backtest's first
    rebalance; the margin makes sure a rate dated before it exists even when
    it follows bond-market holidays and a weekend.

    Caches the rates in input/Risk Free Rate History YYYY-MM-DD.csv (today's
    date) so that repeated calls on the same day do not download again. A
    cached file with no valid rate before first_date (for example, one
    written by an earlier run that needed less history) is ignored and
    replaced. The input/ folder is created when the cache is written, if it
    does not exist yet.

    Parameters
    ----------
    dm : DateManager
        Supplies today's date for the cache file name.

    first_date : pd.Timestamp
        The earliest optimization date. Any time of day is ignored.

    Returns
    -------
    pd.DataFrame
        A DatetimeIndex and a "DTB3" column holding the annualized rate in
        percent, empty on days with no published rate.

    Raises
    ------
    ValueError
        If FRED returns no valid DTB3 rate dated before first_date. Nothing
        is cached then.
    """
    today_date = dm.get_today_date()
    history_filename = os.path.join("input", f"Risk Free Rate History {today_date}.csv")
    if os.path.exists(history_filename):
        print("Reading risk-free rate history cache...")
        tb3m_df = pd.read_csv(history_filename, index_col=0, parse_dates=True)
        if has_rate_before(tb3m_df, first_date):
            return tb3m_df
        print(
            "Cached risk-free rate history has no rate before "
            f"{first_date.date()}; downloading again..."
        )
    end_date = datetime.datetime.now()
    margin = pd.Timedelta(days=RISK_FREE_HISTORY_MARGIN_DAYS)
    start_date = first_date.normalize() - margin
    tb3m_df = web.DataReader("DTB3", "fred", start_date, end_date)
    if not has_rate_before(tb3m_df, first_date):
        raise ValueError(
            "No non-missing DTB3 (3-month T-bill) rate dated before "
            f"{first_date.date()} in the data."
        )
    tb3m_df.to_csv(ensure_parent_dir(history_filename))
    return tb3m_df
