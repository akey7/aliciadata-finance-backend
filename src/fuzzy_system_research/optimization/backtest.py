"""
Walk-forward backtest of the max Sharpe ratio portfolio against an equally
weighted baseline, over the most recent scored trading days, plus writers
for the two files the frontend reads (see docs/INTERFACE_CONTRACT.md).

With t the last session, E scored days, T training returns, and a rebalance
every R sessions:

- The rebalance dates are t-E, t-E+R, ..., t-R, so there are E / R of them,
  and there is no rebalance at t.
- At the close of each rebalance date, the max Sharpe ratio weights are
  estimated from the T daily returns ending on that date (T + 1 closes), with
  the last DTB3 rate published on an earlier date. PortfolioProblem receives
  only that slice of prices, so nothing after the rebalance date can reach
  the estimates or the weights.
- Each portfolio moves to its target weights and is held for R sessions, with
  no trading, so its weights drift with the assets' returns. The baseline
  moves back to 1 / n each on the same dates.
- Wealth starts at $1.00 at the close of t-E and compounds the E daily
  portfolio returns, giving E + 1 observations.

The last rebalance is at t-R, so its weights are not the live portfolio's. The
live optimization at t uses the same training window length and risk-free
rule, and holds nothing in the backtest. portfolio_backtest_metadata.yml
records it as final_optimization, copied from the live result rather than
solved again, and verify_final_optimization() checks that record against
optimization_metadata.yml and portfolio_optimization_plot_data.h5.

Conventions:
- The window lengths come from config/modeling_and_optimization.yml.
- Returns are simple daily returns of adjusted closes. They are fractions in
  this module's results and percent in the written files. Massive.com's
  adjusted closes are not assumed to include dividends, so these may be price
  returns rather than total returns.
- Constraints: the same PortfolioProblem as the live optimization (long-only,
  fully invested, each weight capped). They apply to target weights only;
  drifted weights can exceed the cap between rebalances.
- CAGR is compounded, unlike the arithmetic annualization of
  optimization_metadata.yml: (end wealth)^(252 / E) - 1. It extrapolates a
  short window to a full year, so it is noisy.
- Annualized volatility is realized: the sample standard deviation (ddof = 1)
  of the E daily returns × √252, the same annualization as
  optimization_metadata.yml. It is noisy too, and it needs at least two
  scored days.
- Execution is idealized: rebalancing happens exactly at the close whose
  price the optimization has just used, with no transaction costs or
  slippage. This is not an execution-realistic trading simulation, and the
  universe of tickers is fixed and was chosen with hindsight.
- Fallback: if an optimization does not converge, or its weights violate the
  constraints, the portfolio rebalances back to the previous target weights
  (equal weights for the first rebalance), which use only earlier information
  and are always feasible. Each rebalance records its status.

check_backtest_prices() rejects missing data before anything is computed, and
check_backtest_invariants(), verify_backtest_files(), and
verify_final_optimization() raise InvariantError, so a failed check stops the
run before anything is uploaded.
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable
import h5py
import numpy as np
import pandas as pd
import yaml
from fuzzy_system_research.optimization.portfolio import (
    OptimizationError,
    max_sharpe_weights,
    risk_free_rate_before,
)
from fuzzy_system_research.optimization.problem import (
    TRADING_DAYS_PER_YEAR,
    PortfolioProblem,
    PortfolioStats,
    annualize_risk,
)
from fuzzy_system_research.optimization.workflow import (
    EXACT_RELATIVE_TOLERANCE,
    WEIGHT_TOLERANCE,
    InvariantError,
    OptimizationResult,
)

# RebalanceRecord.status values.
STATUS_OPTIMIZED = "optimized"
STATUS_FALLBACK_PREVIOUS = "fallback_previous_weights"
STATUS_FALLBACK_EQUAL = "fallback_equal_weights"

MAX_SHARPE = "max_sharpe"
EQUAL_WEIGHT = "equal_weight"

# Written to portfolio_backtest_metadata.yml, so the limits of the comparison
# travel with the numbers.
ASSUMPTIONS = {
    "execution": (
        "idealized close-to-close rebalancing with no transaction costs or slippage; "
        "not an execution-realistic trading simulation"
    ),
    "universe": "a fixed set of tickers, chosen with hindsight",
    "prices": (
        "adjusted closes as returned by Massive.com; dividends are not assumed "
        "to be included"
    ),
    "returns": "simple daily returns",
    "risk_free_rate": "the last DTB3 rate dated before each rebalance",
    "drift": "weights drift with returns between rebalances",
    "fallback": (
        "if an optimization fails, rebalance to the previous target weights, or to "
        "equal weights at the first rebalance"
    ),
}


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Performance over the scored days, as fractions (0.05 means 5 %).

    Attributes
    ----------
    cumulative_return : float
        Compounded total return.

    cagr : float
        cumulative_return annualized by compounding.

    max_drawdown : float
        Largest peak-to-trough decline of wealth, including the initial
        $1.00 in the running peak. Zero or negative.

    annualized_volatility : float
        Sample standard deviation of the daily returns (ddof = 1) × √252.
    """

    cumulative_return: float
    cagr: float
    max_drawdown: float
    annualized_volatility: float


@dataclass(frozen=True, eq=False)
class PortfolioPath:
    """
    One portfolio's path through the backtest, with m rebalances and E
    scored days. Weights are fractions, in the order of BacktestResult.tickers.

    Attributes
    ----------
    target_weights : np.ndarray
        Weights set at each rebalance, shape (m, n).

    weights_before_rebalance : np.ndarray
        Drifted weights just before each rebalance, shape (m, n). The first
        row is NaN, because nothing was held before the first rebalance.

    daily_returns : np.ndarray
        Daily portfolio returns, shape (E,).

    wealth : np.ndarray
        Wealth in dollars, starting at 1.0, shape (E + 1,).

    metrics : PerformanceMetrics
        Computed from wealth.
    """

    target_weights: np.ndarray
    weights_before_rebalance: np.ndarray
    daily_returns: np.ndarray
    wealth: np.ndarray
    metrics: PerformanceMetrics


@dataclass(frozen=True)
class RebalanceRecord:
    """
    One rebalance and the information behind it.

    Attributes
    ----------
    date : pd.Timestamp
        The rebalance date (the optimization cutoff).

    first_close_date, first_return_date, last_close_date : pd.Timestamp
        First close, first daily return, and last close of the training
        window. The return on first_return_date uses the close on
        first_close_date.

    n_training_returns : int
        Daily returns in the training window.

    holding_start_date, holding_end_date : pd.Timestamp
        First and last session of the holding period that follows.

    annual_risk_free_rate : float
        DTB3 rate used, annualized, in percent.

    risk_free_rate_date : pd.Timestamp
        Observation date of that rate.

    status : str
        STATUS_OPTIMIZED, STATUS_FALLBACK_PREVIOUS, or STATUS_FALLBACK_EQUAL.

    max_sharpe_stats, equal_weight_stats : PortfolioStats
        Each portfolio's target weights evaluated on this training window:
        in-sample estimates (annualized arithmetically, in percent), not
        realized results.
    """

    date: pd.Timestamp
    first_close_date: pd.Timestamp
    first_return_date: pd.Timestamp
    last_close_date: pd.Timestamp
    n_training_returns: int
    holding_start_date: pd.Timestamp
    holding_end_date: pd.Timestamp
    annual_risk_free_rate: float
    risk_free_rate_date: pd.Timestamp
    status: str
    max_sharpe_stats: PortfolioStats
    equal_weight_stats: PortfolioStats


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """
    Results of one walk-forward backtest.

    Attributes
    ----------
    tickers : tuple[str, ...]
        Assets, in the order of every weight vector.

    dates : pd.DatetimeIndex
        The E + 1 sessions of the wealth series, starting at the first
        rebalance date. Daily returns belong to dates[1:].

    rebalances : tuple[RebalanceRecord, ...]
        One record per rebalance, in date order.

    max_sharpe, equal_weight : PortfolioPath
        The two portfolios.

    training_window_returns, evaluation_days, rebalance_interval : int
        The settings used.

    weight_cap_multiple : int or float
        The setting the weight cap comes from.

    weight_cap : float
        Largest allowed target weight, as a fraction of the portfolio.
    """

    tickers: tuple[str, ...]
    dates: pd.DatetimeIndex
    rebalances: tuple[RebalanceRecord, ...]
    max_sharpe: PortfolioPath
    equal_weight: PortfolioPath
    training_window_returns: int
    evaluation_days: int
    rebalance_interval: int
    weight_cap_multiple: int | float
    weight_cap: float


def wealth_index(daily_returns: np.ndarray) -> np.ndarray:
    """
    Compound daily returns into wealth, starting at $1.00. Pure function.

    Parameters
    ----------
    daily_returns : np.ndarray
        Daily simple returns as fractions, shape (E,).

    Returns
    -------
    np.ndarray
        Wealth in dollars, shape (E + 1,), with 1.0 first.
    """
    return np.concatenate([[1.0], np.cumprod(1.0 + daily_returns)])


def performance_metrics(
    wealth: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> PerformanceMetrics:
    """
    Cumulative return, CAGR, maximum drawdown, and annualized volatility of a
    wealth series. The initial value counts toward the running peak, so a
    loss from the first day is a drawdown. Pure function.

    Parameters
    ----------
    wealth : np.ndarray
        Wealth, shape (E + 1,), including the initial value, with E at least
        2 so the volatility is defined.

    periods_per_year : int, optional
        Trading days per year. Defaults to 252.

    Returns
    -------
    PerformanceMetrics
        The four metrics, as fractions.
    """
    growth = wealth[-1] / wealth[0]
    years = (len(wealth) - 1) / periods_per_year
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1.0
    daily_returns = wealth[1:] / wealth[:-1] - 1.0
    return PerformanceMetrics(
        cumulative_return=float(growth - 1.0),
        cagr=float(growth ** (1.0 / years) - 1.0),
        max_drawdown=float(drawdowns.min()),
        annualized_volatility=float(
            annualize_risk(daily_returns.std(ddof=1), periods_per_year)
        ),
    )


def drift_through_holding(
    weights: np.ndarray, asset_returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hold a portfolio without trading: each position compounds at its asset's
    return, so the weights drift. Pure function.

    Parameters
    ----------
    weights : np.ndarray
        Weights at the start of the holding period, shape (n,).

    asset_returns : np.ndarray
        Daily simple returns as fractions, shape (days, n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The daily portfolio returns, shape (days,), and the drifted weights
        at the end of the holding period, shape (n,).
    """
    positions = weights * np.cumprod(1.0 + asset_returns, axis=0)
    values = np.concatenate([[weights.sum()], positions.sum(axis=1)])
    return values[1:] / values[:-1] - 1.0, positions[-1] / values[-1]


def simulate_portfolio(
    target_weights: np.ndarray, holding_returns: list[np.ndarray]
) -> PortfolioPath:
    """
    Run one portfolio through consecutive holding periods, moving to the
    next target weights at the end of each. Pure function.

    Parameters
    ----------
    target_weights : np.ndarray
        Weights set at each rebalance, shape (m, n).

    holding_returns : list[np.ndarray]
        Asset returns of each of the m holding periods, as fractions.

    Returns
    -------
    PortfolioPath
        Its weights, daily returns, wealth, and metrics.
    """
    weights_before_rebalance = np.full(target_weights.shape, np.nan)
    daily_returns = []
    for i, (weights, asset_returns) in enumerate(zip(target_weights, holding_returns)):
        period_returns, drifted = drift_through_holding(weights, asset_returns)
        daily_returns.append(period_returns)
        if i + 1 < len(target_weights):
            weights_before_rebalance[i + 1] = drifted
    returns = np.concatenate(daily_returns)
    wealth = wealth_index(returns)
    return PortfolioPath(
        target_weights=target_weights,
        weights_before_rebalance=weights_before_rebalance,
        daily_returns=returns,
        wealth=wealth,
        metrics=performance_metrics(wealth),
    )


def check_backtest_prices(
    wide_df: pd.DataFrame, n_sessions: int, expected_sessions: pd.DatetimeIndex
) -> None:
    """
    Require a complete price table: exactly n_sessions rows, on exactly the
    expected consecutive trading sessions, with no missing close. A backtest
    with a gap would silently use a different number of returns.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide adjusted close prices (tickers in columns, sorted DatetimeIndex).

    n_sessions : int
        Number of sessions the backtest needs.

    expected_sessions : pd.DatetimeIndex
        The trading sessions the table must cover.

    Raises
    ------
    ValueError
        Naming what is missing.
    """
    if len(wide_df) != n_sessions:
        raise ValueError(
            f"The backtest needs exactly {n_sessions} sessions of prices, got "
            f"{len(wide_df)}."
        )
    actual = wide_df.index.normalize()
    expected = pd.DatetimeIndex(expected_sessions).normalize()
    if not actual.equals(expected):
        missing = [str(day.date()) for day in expected.difference(actual)]
        raise ValueError(
            f"The backtest's prices are not the expected consecutive sessions; "
            f"missing: {missing}."
        )
    if wide_df.isna().any(axis=None):
        gaps = wide_df.isna().stack()
        first_date, ticker = gaps[gaps].index[0]
        raise ValueError(
            f"The backtest needs every close, but {ticker} has none on "
            f"{first_date.date()} ({int(gaps.sum())} missing in all)."
        )


def _target_weights(
    problem: PortfolioProblem,
    optimizer: Callable[[PortfolioProblem], np.ndarray],
    previous: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """
    The weights to rebalance to and their RebalanceRecord.status: the
    optimizer's, tidied of rounding noise (no negative weights, summing to
    exactly 1), or the fallback described in the module docstring.
    """
    try:
        weights = np.asarray(optimizer(problem), dtype=float)
        if not problem.max_constraint_violation(weights) <= WEIGHT_TOLERANCE:
            raise OptimizationError("The optimized weights violate the constraints.")
    except OptimizationError as error:
        print(f"Backtest optimization failed, using the fallback: {error}")
        if previous is None:
            return problem.equal_weights(), STATUS_FALLBACK_EQUAL
        return previous, STATUS_FALLBACK_PREVIOUS
    clipped = np.clip(weights, 0.0, None)
    return clipped / clipped.sum(), STATUS_OPTIMIZED


def run_portfolio_backtest(
    wide_df: pd.DataFrame,
    tb3m_df: pd.DataFrame,
    *,
    training_window_returns: int,
    evaluation_days: int,
    rebalance_interval: int,
    weight_cap_multiple: int | float,
    expected_sessions: pd.DatetimeIndex,
    optimizer: Callable[[PortfolioProblem], np.ndarray] = max_sharpe_weights,
) -> BacktestResult:
    """
    Run the walk-forward backtest described in the module docstring. Checks
    the prices first and the invariants before returning. Pure computation
    apart from printing a used fallback; does no file or network I/O.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide adjusted close prices, exactly training_window_returns + 1 +
        evaluation_days sessions, as returned by data.prices.last_n_sessions().

    tb3m_df : pd.DataFrame
        FRED DTB3 history: a DatetimeIndex and a "DTB3" column holding the
        annualized rate in percent.

    training_window_returns, evaluation_days, rebalance_interval : int
        Daily returns per optimization, days scored, and sessions between
        rebalances. rebalance_interval must divide evaluation_days.

    weight_cap_multiple : int or float
        Each target weight is at most this / (number of tickers).

    expected_sessions : pd.DatetimeIndex
        The trading sessions wide_df must cover.

    optimizer : callable, optional
        Returns weights for a PortfolioProblem. Defaults to
        portfolio.max_sharpe_weights; tests pass doubles.

    Returns
    -------
    BacktestResult
        Both portfolios' paths and the record of each rebalance.

    Raises
    ------
    ValueError
        If evaluation_days is below 2, rebalance_interval does not divide
        evaluation_days, a price is missing, or no DTB3 rate precedes a
        rebalance date.

    InvariantError
        If the result violates check_backtest_invariants().
    """
    if evaluation_days < 2:
        raise ValueError(
            f"The backtest needs at least 2 scored days for its volatility, got "
            f"{evaluation_days}."
        )
    if evaluation_days % rebalance_interval != 0:
        raise ValueError(
            f"{evaluation_days} scored days do not fall into whole holding periods "
            f"of {rebalance_interval} sessions."
        )
    n_sessions = training_window_returns + 1 + evaluation_days
    check_backtest_prices(wide_df, n_sessions, expected_sessions)

    records: list[RebalanceRecord] = []
    max_sharpe_targets: list[np.ndarray] = []
    holding_returns: list[np.ndarray] = []
    previous = None
    weight_cap = None
    # The first rebalance is the close at which the first training window
    # ends; the last is one holding period before the last session.
    for position in range(
        training_window_returns, n_sessions - 1, rebalance_interval
    ):
        date = wide_df.index[position]
        training_prices = wide_df.iloc[position - training_window_returns : position + 1]
        rate = risk_free_rate_before(tb3m_df, date)
        problem = PortfolioProblem.from_prices(
            training_prices, rate["daily_risk_free_rate"], weight_cap_multiple
        )
        weight_cap = problem.weight_cap
        weights, status = _target_weights(problem, optimizer, previous)
        previous = weights
        max_sharpe_targets.append(weights)

        holding_prices = wide_df.iloc[position : position + rebalance_interval + 1]
        holding_returns.append(holding_prices.pct_change().iloc[1:].to_numpy())
        records.append(
            RebalanceRecord(
                date=date,
                first_close_date=training_prices.index[0],
                first_return_date=training_prices.index[1],
                last_close_date=training_prices.index[-1],
                n_training_returns=len(training_prices) - 1,
                holding_start_date=holding_prices.index[1],
                holding_end_date=holding_prices.index[-1],
                annual_risk_free_rate=rate["risk_free_rate"],
                risk_free_rate_date=pd.Timestamp(rate["risk_free_rate_date"]),
                status=status,
                max_sharpe_stats=problem.portfolio_stats(weights),
                equal_weight_stats=problem.portfolio_stats(problem.equal_weights()),
            )
        )

    targets = np.array(max_sharpe_targets)
    n_assets = wide_df.shape[1]
    result = BacktestResult(
        tickers=tuple(wide_df.columns),
        dates=wide_df.index[training_window_returns:],
        rebalances=tuple(records),
        max_sharpe=simulate_portfolio(targets, holding_returns),
        equal_weight=simulate_portfolio(
            np.full(targets.shape, 1.0 / n_assets), holding_returns
        ),
        training_window_returns=training_window_returns,
        evaluation_days=evaluation_days,
        rebalance_interval=rebalance_interval,
        weight_cap_multiple=weight_cap_multiple,
        weight_cap=weight_cap,
    )
    check_backtest_invariants(result)
    return result


def _fail(message: str) -> None:
    raise InvariantError(f"Portfolio backtest check failed: {message}")


def _same(a: float, b: float) -> bool:
    return bool(
        np.isclose(
            a, b, rtol=EXACT_RELATIVE_TOLERANCE, atol=EXACT_RELATIVE_TOLERANCE
        )
    )


def check_backtest_invariants(result: BacktestResult) -> None:
    """
    Check the backtest's structure, and raise InvariantError (stopping the
    run before anything is uploaded) if it does not hold.

    Checks:
    1. There is one rebalance per holding period, and evaluation_days + 1
       dates.
    2. Every training window holds exactly training_window_returns returns,
       ends on its rebalance date, and uses a risk-free rate dated before it.
    3. The holding periods are consecutive, do not overlap, each last
       rebalance_interval sessions, and exactly cover dates[1:].
    4. Each holding period starts after its training window ends, so no
       training window holds a return from its own holding period.
    5. Every target weight satisfies the constraints.
    6. Each portfolio has evaluation_days returns, and its wealth starts at
       1.0 and compounds them.
    7. Each portfolio's metrics equal those recomputed from its wealth.

    Parameters
    ----------
    result : BacktestResult
        The result to check.
    """
    n_rebalances = result.evaluation_days // result.rebalance_interval
    if len(result.rebalances) != n_rebalances:
        _fail(f"expected {n_rebalances} rebalances, got {len(result.rebalances)}.")
    if len(result.dates) != result.evaluation_days + 1:
        _fail(f"expected {result.evaluation_days + 1} dates, got {len(result.dates)}.")

    for i, record in enumerate(result.rebalances):
        start = i * result.rebalance_interval
        holding = result.dates[start + 1 : start + result.rebalance_interval + 1]
        if record.n_training_returns != result.training_window_returns:
            _fail(f"the {record.date.date()} window has {record.n_training_returns} returns.")
        if record.date != result.dates[start] or record.last_close_date != record.date:
            _fail(f"the training window of {record.date.date()} ends on the wrong date.")
        if not record.risk_free_rate_date.normalize() < record.date.normalize():
            _fail(f"the {record.date.date()} risk-free rate is not dated before it.")
        if len(holding) != result.rebalance_interval or (
            record.holding_start_date != holding[0]
            or record.holding_end_date != holding[-1]
        ):
            _fail(f"the holding period after {record.date.date()} is wrong.")
        if not record.last_close_date < record.holding_start_date:
            _fail(f"the training window of {record.date.date()} overlaps its holding period.")

    for name, path in ((MAX_SHARPE, result.max_sharpe), (EQUAL_WEIGHT, result.equal_weight)):
        weights = path.target_weights
        violation = max(
            (-weights).max(),
            (weights - result.weight_cap).max(),
            np.abs(weights.sum(axis=1) - 1.0).max(),
        )
        if not violation <= WEIGHT_TOLERANCE:
            _fail(f"{name} target weights violate the constraints by {violation:.3g}.")
        if len(path.daily_returns) != result.evaluation_days:
            _fail(f"{name} has {len(path.daily_returns)} daily returns.")
        if len(path.wealth) != result.evaluation_days + 1 or path.wealth[0] != 1.0:
            _fail(f"{name} wealth does not start at 1.0 with one value per date.")
        if not np.allclose(
            path.wealth,
            wealth_index(path.daily_returns),
            rtol=EXACT_RELATIVE_TOLERANCE,
            atol=EXACT_RELATIVE_TOLERANCE,
        ):
            _fail(f"{name} wealth does not compound its daily returns.")
        recomputed = performance_metrics(path.wealth)
        if not (
            _same(recomputed.cumulative_return, path.metrics.cumulative_return)
            and _same(recomputed.cagr, path.metrics.cagr)
            and _same(recomputed.max_drawdown, path.metrics.max_drawdown)
            and _same(recomputed.annualized_volatility, path.metrics.annualized_volatility)
        ):
            _fail(f"{name} metrics differ from those of its wealth.")


def _date_str(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%d")


def _weights_pct(tickers: tuple[str, ...], weights: np.ndarray) -> dict[str, float] | None:
    # Plain floats, so yaml.dump writes numbers rather than numpy tags. None
    # for the NaN row before the first rebalance.
    if np.isnan(weights).any():
        return None
    return {ticker: float(w * 100) for ticker, w in zip(tickers, weights)}


def backtest_daily_frame(result: BacktestResult) -> pd.DataFrame:
    """
    The table written to portfolio_backtest_daily.csv: one row per date,
    daily returns in percent (empty on the first row, the initial wealth) and
    wealth in dollars for both portfolios. Pure function.

    Parameters
    ----------
    result : BacktestResult
        The backtest.

    Returns
    -------
    pd.DataFrame
        Indexed by "date" (YYYY-MM-DD strings).
    """

    def percent(path: PortfolioPath) -> np.ndarray:
        return np.concatenate([[np.nan], path.daily_returns * 100])

    return pd.DataFrame(
        {
            f"{MAX_SHARPE}_daily_return": percent(result.max_sharpe),
            f"{MAX_SHARPE}_wealth": result.max_sharpe.wealth,
            f"{EQUAL_WEIGHT}_daily_return": percent(result.equal_weight),
            f"{EQUAL_WEIGHT}_wealth": result.equal_weight.wealth,
        },
        index=pd.Index(result.dates.strftime("%Y-%m-%d"), name="date"),
    )


def build_final_optimization(
    optimization: OptimizationResult,
    risk_free_rate_data: dict,
    training_dates: pd.DatetimeIndex,
) -> dict:
    """
    Build the final_optimization section of portfolio_backtest_metadata.yml
    from the live optimization, without solving anything again, so it holds
    exactly the values of optimization_metadata.yml. Pure function.

    Parameters
    ----------
    optimization : OptimizationResult
        The live optimization's result.

    risk_free_rate_data : dict
        The rate it used, as returned by portfolio.risk_free_rate_before().

    training_dates : pd.DatetimeIndex
        The sessions of the closes it was given.

    Returns
    -------
    dict
        The section, ready for build_backtest_metadata().
    """
    return {
        "date": _date_str(optimization.date_to),
        "training_window": {
            "first_close_date": _date_str(training_dates[0]),
            "first_return_date": _date_str(training_dates[1]),
            "last_close_date": _date_str(training_dates[-1]),
            "returns": len(training_dates) - 1,
        },
        "risk_free_rate": {
            "risk_free_rate": risk_free_rate_data["risk_free_rate"],
            "risk_free_rate_date": risk_free_rate_data["risk_free_rate_date"][:10],
        },
        "annualized_return": optimization.max_sharpe_return,
        "risk": optimization.max_sharpe_risk,
        "sharpe_ratio": optimization.sharpe_ratio,
        "weights": optimization.max_sharpe_weights_pct,
    }


def build_backtest_metadata(
    result: BacktestResult, tickers: list[str], final_optimization: dict
) -> dict:
    """
    Build the contents of portfolio_backtest_metadata.yml as specified in
    docs/INTERFACE_CONTRACT.md. Metrics and weights are in percent. Pure
    function.

    Parameters
    ----------
    result : BacktestResult
        The backtest.

    tickers : list[str]
        Tickers of the portfolio, in the backend's own order.

    final_optimization : dict
        As returned by build_final_optimization().

    Returns
    -------
    dict
        Nested dictionary ready for yaml.dump().
    """

    def metrics_pct(path: PortfolioPath) -> dict[str, float]:
        return {
            "cumulative_return": path.metrics.cumulative_return * 100,
            "cagr": path.metrics.cagr * 100,
            "max_drawdown": path.metrics.max_drawdown * 100,
            "annualized_volatility": path.metrics.annualized_volatility * 100,
        }

    def portfolio_at(path: PortfolioPath, i: int, stats: PortfolioStats) -> dict:
        return {
            "target_weights": _weights_pct(result.tickers, path.target_weights[i]),
            "weights_before_rebalance": _weights_pct(
                result.tickers, path.weights_before_rebalance[i]
            ),
            "in_sample": dataclasses.asdict(stats),
        }

    rebalances = [
        {
            "date": _date_str(record.date),
            "status": record.status,
            "training_window": {
                "first_close_date": _date_str(record.first_close_date),
                "first_return_date": _date_str(record.first_return_date),
                "last_close_date": _date_str(record.last_close_date),
                "returns": record.n_training_returns,
            },
            "holding_period": {
                "start_date": _date_str(record.holding_start_date),
                "end_date": _date_str(record.holding_end_date),
            },
            "risk_free_rate": {
                "risk_free_rate": record.annual_risk_free_rate,
                "risk_free_rate_date": _date_str(record.risk_free_rate_date),
            },
            MAX_SHARPE: portfolio_at(result.max_sharpe, i, record.max_sharpe_stats),
            EQUAL_WEIGHT: portfolio_at(result.equal_weight, i, record.equal_weight_stats),
        }
        for i, record in enumerate(result.rebalances)
    ]
    return {
        "tickers": list(tickers),
        "evaluation_window": {
            "initial_wealth_date": _date_str(result.dates[0]),
            "start_date": _date_str(result.dates[1]),
            "end_date": _date_str(result.dates[-1]),
            "trading_days": result.evaluation_days,
        },
        "schedule": {
            "training_window_returns": result.training_window_returns,
            "rebalance_interval_sessions": result.rebalance_interval,
            "rebalance_count": len(result.rebalances),
            "fallback_count": sum(
                record.status != STATUS_OPTIMIZED for record in result.rebalances
            ),
        },
        "constraints": {
            "long_only": True,
            "fully_invested": True,
            "weight_cap": float(result.weight_cap * 100),
            "weight_cap_multiple": result.weight_cap_multiple,
        },
        "annualization": {
            "method": "compounded",
            "periods_per_year": TRADING_DAYS_PER_YEAR,
        },
        "units": {
            "daily_return": "percent",
            "wealth": "usd",
            "cumulative_return": "percent",
            "cagr": "percent",
            "max_drawdown": "percent",
            "annualized_volatility": "percent_per_year",
            "weights": "percent",
            "risk_free_rate": "percent_per_year",
        },
        "assumptions": dict(ASSUMPTIONS),
        "metrics": {
            MAX_SHARPE: metrics_pct(result.max_sharpe),
            EQUAL_WEIGHT: metrics_pct(result.equal_weight),
        },
        "rebalances": rebalances,
        "final_optimization": final_optimization,
    }


def write_backtest_daily(result: BacktestResult, path: str) -> None:
    """
    Write portfolio_backtest_daily.csv as returned by
    backtest_daily_frame(). Overwrites the file.

    Parameters
    ----------
    result : BacktestResult
        The backtest.

    path : str
        Local path of the .csv file to write. Its folder must exist.
    """
    backtest_daily_frame(result).to_csv(path, index=True)


def write_backtest_metadata(metadata: dict, path: str) -> None:
    """
    Write portfolio_backtest_metadata.yml (block style, keys sorted).
    Overwrites the file.

    Parameters
    ----------
    metadata : dict
        As returned by build_backtest_metadata().

    path : str
        Local path of the .yml file to write. Its folder must exist.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def verify_backtest_files(daily_path: str, metadata_path: str) -> None:
    """
    Read back the two written backtest files and check that they agree,
    raising InvariantError (before anything is uploaded) if they do not: the
    daily file has trading_days + 1 rows over the evaluation window, each
    wealth series starts at 1.0 and compounds its daily returns, the
    rebalance count matches the schedule, and the metrics in the .yml file
    equal those recomputed from the wealth in the .csv file.

    Parameters
    ----------
    daily_path : str
        Local path of portfolio_backtest_daily.csv.

    metadata_path : str
        Local path of portfolio_backtest_metadata.yml.
    """
    daily = pd.read_csv(daily_path, index_col="date")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    window = metadata["evaluation_window"]
    if len(daily) != window["trading_days"] + 1:
        _fail("the .csv file's row count differs from trading_days + 1 in the .yml file.")
    if (daily.index[0], daily.index[1], daily.index[-1]) != (
        window["initial_wealth_date"],
        window["start_date"],
        window["end_date"],
    ):
        _fail("the .csv file's dates differ from evaluation_window in the .yml file.")
    if len(metadata["rebalances"]) != metadata["schedule"]["rebalance_count"]:
        _fail("the .yml file's rebalances differ in number from rebalance_count.")
    for name in (MAX_SHARPE, EQUAL_WEIGHT):
        wealth = daily[f"{name}_wealth"].to_numpy()
        returns = daily[f"{name}_daily_return"].to_numpy()[1:] / 100
        if wealth[0] != 1.0 or not np.allclose(
            wealth,
            wealth_index(returns),
            rtol=EXACT_RELATIVE_TOLERANCE,
            atol=EXACT_RELATIVE_TOLERANCE,
        ):
            _fail(f"the .csv file's {name} wealth does not compound its daily returns.")
        recomputed = performance_metrics(wealth)
        written = metadata["metrics"][name]
        if not (
            _same(recomputed.cumulative_return * 100, written["cumulative_return"])
            and _same(recomputed.cagr * 100, written["cagr"])
            and _same(recomputed.max_drawdown * 100, written["max_drawdown"])
            and _same(recomputed.annualized_volatility * 100, written["annualized_volatility"])
        ):
            _fail(f"the .yml file's {name} metrics differ from the .csv file's wealth.")


def verify_final_optimization(
    backtest_metadata_path: str, optimization_metadata_path: str, plot_data_path: str
) -> None:
    """
    Read back the written files and check that final_optimization in
    portfolio_backtest_metadata.yml is the live optimization, raising
    InvariantError (before anything is uploaded) if it is not: it ends on
    evaluation_window.end_date and date_updated.date_to, starts on
    date_updated.date_from, has schedule.training_window_returns returns,
    and its risk-free rate, return, volatility, Sharpe ratio, and weights
    equal those of optimization_metadata.yml, and its return and volatility
    equal the max Sharpe ratio point in the .h5 file.

    Parameters
    ----------
    backtest_metadata_path : str
        Local path of portfolio_backtest_metadata.yml.

    optimization_metadata_path : str
        Local path of optimization_metadata.yml.

    plot_data_path : str
        Local path of portfolio_optimization_plot_data.h5.
    """
    with open(backtest_metadata_path, encoding="utf-8") as f:
        backtest = yaml.safe_load(f)
    with open(optimization_metadata_path, encoding="utf-8") as f:
        optimization = yaml.safe_load(f)
    with h5py.File(plot_data_path, "r") as hf:
        h5_risk = float(hf["max_sharpe_ratio/xs"][0])
        h5_return = float(hf["max_sharpe_ratio/ys"][0])

    final = backtest["final_optimization"]
    window = final["training_window"]
    dates = optimization["date_updated"]
    if not (
        final["date"]
        == window["last_close_date"]
        == dates["date_to"]
        == backtest["evaluation_window"]["end_date"]
    ):
        _fail("final_optimization does not end on the optimization's last date.")
    if window["first_close_date"] != dates["date_from"]:
        _fail("final_optimization does not start on the optimization's first date.")
    if window["returns"] != backtest["schedule"]["training_window_returns"]:
        _fail("final_optimization has a different training window from the rebalances.")

    rate, live_rate = final["risk_free_rate"], optimization["risk_free_rate"]
    if not (
        _same(rate["risk_free_rate"], live_rate["risk_free_rate"])
        and rate["risk_free_rate_date"] == live_rate["risk_free_rate_date"][:10]
    ):
        _fail("final_optimization has a different risk-free rate from the optimization.")

    live = optimization["optimum_portfolio"]
    if not all(
        _same(final[key], live[key]) for key in ("annualized_return", "risk", "sharpe_ratio")
    ):
        _fail("final_optimization's statistics differ from optimum_portfolio.")
    if final["weights"].keys() != live["weights"].keys() or not all(
        _same(weight, live["weights"][ticker]) for ticker, weight in final["weights"].items()
    ):
        _fail("final_optimization's weights differ from optimum_portfolio.")
    if not (_same(final["risk"], h5_risk) and _same(final["annualized_return"], h5_return)):
        _fail("final_optimization differs from the .h5 max Sharpe ratio point.")
