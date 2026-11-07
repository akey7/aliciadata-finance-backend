"""
Settings for the backend's models and optimization, read from a YAML file
(by default config/modeling_and_optimization.yml, relative to the working
directory, like input/ and output/).

The file is read once by the entry script, which passes the parsed settings
down explicitly; nothing reads it at import time. Validation is strict:
every key is required and unknown keys are rejected, so the file is the one
source of truth and a typo fails the run instead of falling back to a
default.

File layout:

    time_series_models:
      training_window_length: <int>    # closes per ARIMA training window
      evaluation_window_length: <int>  # backtest days scored
    portfolio_optimization:
      weight_cap_multiple: <number>     # cap = multiple / number of tickers
      efficient_frontier_points: <int>  # target returns on the frontier
      simulated_portfolios: <int>       # random feasible portfolios
      random_seed: <int>                # seed for the random portfolios
      near_frontier_tolerance: <number> # pp of return counted as "near" the frontier
    portfolio_backtest:
      training_window_returns: <int>     # daily returns per optimization, live too
      rebalance_interval_sessions: <int> # trading sessions between rebalances

The backtest scores the same number of days as the forecasts
(time_series_models.evaluation_window_length) and uses the same weight cap as
the optimization (portfolio_optimization.weight_cap_multiple).

The current values are in the file itself.
"""

import math
import os
from dataclasses import dataclass
import yaml

DEFAULT_CONFIG_PATH = os.path.join("config", "modeling_and_optimization.yml")
# Smallest training window accepted, in closes. ARIMA(1, 0, 1) with a
# constant estimates four parameters, so far fewer returns than this would
# make the fits meaningless.
MIN_TRAINING_WINDOW_LENGTH = 30
# Smallest backtest training window accepted, in daily returns. Far fewer than
# this would leave the covariance matrix of the tickers singular or meaningless.
MIN_BACKTEST_TRAINING_RETURNS = 30


class ConfigError(ValueError):
    """The configuration file is missing, unreadable, or invalid."""


def _require_int(
    name: str, value: object, minimum: int, section: str = "time_series_models"
) -> None:
    # bool is a subclass of int, so `True` must be rejected explicitly.
    if type(value) is not int:
        raise ConfigError(f"{section}.{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ConfigError(f"{section}.{name} must be at least {minimum}, got {value}.")


def _require_number_above(name: str, value: object, lower: float, section: str) -> None:
    # An int or a float (not a bool), finite and strictly greater than lower.
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{section}.{name} must be a number, got {value!r}.")
    if not value > lower:
        raise ConfigError(f"{section}.{name} must be greater than {lower}, got {value}.")


@dataclass(frozen=True)
class TimeSeriesConfig:
    """
    Settings for the walk-forward time series models. Validated on
    construction.

    Attributes
    ----------
    training_window_length : int
        Closing prices (NYSE trading sessions) in each training window,
        ending on the session before the forecast date. At least
        MIN_TRAINING_WINDOW_LENGTH.

    evaluation_window_length : int
        Most recent trading days with an actual close that are backtested
        and scored: one forecast each, plus the live forecast. At least 1.
    """

    training_window_length: int
    evaluation_window_length: int

    def __post_init__(self) -> None:
        _require_int(
            "training_window_length",
            self.training_window_length,
            MIN_TRAINING_WINDOW_LENGTH,
        )
        _require_int("evaluation_window_length", self.evaluation_window_length, 1)

    @property
    def training_window_returns(self) -> int:
        """Daily log returns in each training window: one fewer than closes."""
        return self.training_window_length - 1

    @property
    def arima_window_sessions(self) -> int:
        """
        Most recent trading sessions the walk-forward ARIMA needs: one
        training window's closes plus one session per scored day, giving
        evaluation_window_length forecasts with actuals plus the live
        forecast: training_window_length + evaluation_window_length.
        """
        return self.training_window_length + self.evaluation_window_length


PORTFOLIO_SECTION = "portfolio_optimization"


@dataclass(frozen=True)
class PortfolioOptimizationConfig:
    """
    Settings for the portfolio optimization and its plot data. Validated on
    construction.

    Attributes
    ----------
    weight_cap_multiple : int or float
        Each weight is at most weight_cap_multiple / (number of tickers),
        for both optimized portfolios, the efficient frontier, and the
        random portfolios. Greater than 1: at 1 the only feasible portfolio
        is equal weights.

    efficient_frontier_points : int
        Target returns on the efficient frontier. At least 2.

    simulated_portfolios : int
        Random feasible portfolios drawn for the plot. At least 1.

    random_seed : int
        Seed of the random number generator for those portfolios. At least 0.

    near_frontier_tolerance : int or float
        A random portfolio counts as near the efficient frontier when its
        annualized return is within this many percentage points of the
        frontier's return at the same volatility. Reported in
        optimization_metadata.yml. Greater than 0.
    """

    weight_cap_multiple: int | float
    efficient_frontier_points: int
    simulated_portfolios: int
    random_seed: int
    near_frontier_tolerance: int | float

    def __post_init__(self) -> None:
        _require_number_above(
            "weight_cap_multiple", self.weight_cap_multiple, 1, PORTFOLIO_SECTION
        )
        _require_int(
            "efficient_frontier_points",
            self.efficient_frontier_points,
            2,
            PORTFOLIO_SECTION,
        )
        _require_int(
            "simulated_portfolios", self.simulated_portfolios, 1, PORTFOLIO_SECTION
        )
        _require_int("random_seed", self.random_seed, 0, PORTFOLIO_SECTION)
        _require_number_above(
            "near_frontier_tolerance", self.near_frontier_tolerance, 0, PORTFOLIO_SECTION
        )


BACKTEST_SECTION = "portfolio_backtest"


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """
    Settings for the walk-forward backtest of the max Sharpe ratio portfolio
    against equal weights. Validated on construction.

    Attributes
    ----------
    training_window_returns : int
        Daily simple returns in each optimization's training window, ending
        on the rebalance date. The live optimization uses the same length,
        so the two can be compared. The window holds one more close than
        this. At least MIN_BACKTEST_TRAINING_RETURNS.

    rebalance_interval_sessions : int
        Trading sessions between rebalances, and so in each holding period.
        At least 1. ModelingConfig checks that it divides the number of
        scored days exactly.
    """

    training_window_returns: int
    rebalance_interval_sessions: int

    def __post_init__(self) -> None:
        _require_int(
            "training_window_returns",
            self.training_window_returns,
            MIN_BACKTEST_TRAINING_RETURNS,
            BACKTEST_SECTION,
        )
        _require_int(
            "rebalance_interval_sessions",
            self.rebalance_interval_sessions,
            1,
            BACKTEST_SECTION,
        )

    @property
    def training_window_closes(self) -> int:
        """Closes in each training window: one more than daily returns."""
        return self.training_window_returns + 1

    def sessions_needed(self, evaluation_days: int) -> int:
        """
        Most recent trading sessions the backtest needs: one training
        window's closes, ending on the first rebalance date, plus one
        session per scored day: training_window_closes + evaluation_days.
        """
        return self.training_window_closes + evaluation_days


@dataclass(frozen=True)
class ModelingConfig:
    """
    The whole configuration file: one field per section. Validated on
    construction: the backtest's scored days must fall into whole holding
    periods.
    """

    time_series_models: TimeSeriesConfig
    portfolio_optimization: PortfolioOptimizationConfig
    portfolio_backtest: PortfolioBacktestConfig

    def __post_init__(self) -> None:
        evaluation_days = self.time_series_models.evaluation_window_length
        interval = self.portfolio_backtest.rebalance_interval_sessions
        if evaluation_days % interval != 0:
            raise ConfigError(
                f"time_series_models.evaluation_window_length ({evaluation_days}) must "
                f"be a multiple of {BACKTEST_SECTION}.rebalance_interval_sessions "
                f"({interval})."
            )


def _check_keys(section: str, mapping: object, expected: set[str]) -> dict:
    if not isinstance(mapping, dict):
        raise ConfigError(f"{section} must be a mapping, got {mapping!r}.")
    # YAML keys need not be strings, so sort and join them as strings.
    missing = sorted(map(str, expected - set(mapping)))
    unknown = sorted(map(str, set(mapping) - expected))
    if missing:
        raise ConfigError(f"{section} is missing required key(s): {', '.join(missing)}.")
    if unknown:
        raise ConfigError(f"{section} has unknown key(s): {', '.join(unknown)}.")
    return mapping


def parse_config(raw: object) -> ModelingConfig:
    """
    Validate a parsed YAML document and build the configuration. Pure
    function.

    Parameters
    ----------
    raw : object
        The document as returned by yaml.safe_load().

    Returns
    -------
    ModelingConfig
        The validated settings.

    Raises
    ------
    ConfigError
        If a section or key is missing or unknown, or a value has the wrong
        type or is out of range.
    """
    top = _check_keys(
        "The configuration file",
        raw,
        {"time_series_models", PORTFOLIO_SECTION, BACKTEST_SECTION},
    )
    ts = _check_keys(
        "time_series_models",
        top["time_series_models"],
        {"training_window_length", "evaluation_window_length"},
    )
    po = _check_keys(
        PORTFOLIO_SECTION,
        top[PORTFOLIO_SECTION],
        {
            "weight_cap_multiple",
            "efficient_frontier_points",
            "simulated_portfolios",
            "random_seed",
            "near_frontier_tolerance",
        },
    )
    bt = _check_keys(
        BACKTEST_SECTION,
        top[BACKTEST_SECTION],
        {"training_window_returns", "rebalance_interval_sessions"},
    )
    return ModelingConfig(
        time_series_models=TimeSeriesConfig(
            training_window_length=ts["training_window_length"],
            evaluation_window_length=ts["evaluation_window_length"],
        ),
        portfolio_optimization=PortfolioOptimizationConfig(
            weight_cap_multiple=po["weight_cap_multiple"],
            efficient_frontier_points=po["efficient_frontier_points"],
            simulated_portfolios=po["simulated_portfolios"],
            random_seed=po["random_seed"],
            near_frontier_tolerance=po["near_frontier_tolerance"],
        ),
        portfolio_backtest=PortfolioBacktestConfig(
            training_window_returns=bt["training_window_returns"],
            rebalance_interval_sessions=bt["rebalance_interval_sessions"],
        ),
    )


def load_config(path: str = DEFAULT_CONFIG_PATH) -> ModelingConfig:
    """
    Read and validate the configuration file.

    Parameters
    ----------
    path : str, optional
        Path of the YAML file, relative to the working directory. Defaults
        to DEFAULT_CONFIG_PATH.

    Returns
    -------
    ModelingConfig
        The validated settings.

    Raises
    ------
    ConfigError
        If the file is missing, is not valid YAML, or fails parse_config().
        The message names the file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Configuration file not found: {path} (run from the repository root, "
            "or pass --config)."
        ) from None
    except yaml.YAMLError as error:
        raise ConfigError(f"Configuration file {path} is not valid YAML: {error}") from None
    try:
        return parse_config(raw)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None
