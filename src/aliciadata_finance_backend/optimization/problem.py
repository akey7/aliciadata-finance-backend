"""
The portfolio optimization problem, defined once: the assets, the daily
return estimates, the risk-free rate, the constraint set, and how a
portfolio's return, volatility, and Sharpe ratio are computed and annualized.

Every optimization and every reported point uses this module, so the
optimizer, the efficient frontier, the random portfolios, and every
coordinate written to portfolio_optimization_plot_data.h5 and
optimization_metadata.yml share one definition and cannot drift apart:

- PortfolioProblem.bounds() and constraints() are the constraint set of the
  max Sharpe ratio, minimum variance, maximum return, and efficient-frontier
  optimizations (see portfolio.py).
- PortfolioProblem.daily_mean_return() and daily_variance() are the only
  implementations of w·μ and wᵀΣw. The optimizers' objectives use them, and
  so does portfolio_stats().
- PortfolioProblem.portfolio_stats() is the only place that turns weights
  into the annualized return, volatility, and Sharpe ratio that are uploaded.
- PortfolioProblem.sample_feasible_weights() draws random portfolios from
  the same constraint set.

Conventions:
- Returns are simple daily returns of adjusted closes, in percent. The mean
  vector and the sample covariance matrix (ddof = 1) are estimated from the
  prices given to from_prices() (about one year of daily data in the
  pipeline), so they are noisy.
- The risk-free rate is daily, in percent (the annual 3-month T-bill rate /
  TRADING_DAYS_PER_YEAR).
- Annualization is arithmetic: return × 252 and volatility × √252, so the
  annualized Sharpe ratio, (annual return − annual risk-free rate) / annual
  volatility, equals the daily Sharpe ratio × √252.
- Constraints: long-only (each weight ≥ 0), fully invested (weights sum to
  1), and each weight at most weight_cap_multiple / (number of assets). The
  multiple is a setting in config/modeling_and_optimization.yml.

Pure computation: no file or network I/O.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# sample_feasible_weights() gives up after this many draws per requested
# portfolio, so a constraint set that rejection sampling can hardly ever hit
# fails loudly instead of looping for a very long time.
MAX_SAMPLER_DRAWS_PER_PORTFOLIO = 1000


def annualize_return(
    daily_return: float | np.ndarray | list[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray | np.float64:
    """
    Arithmetically annualize mean daily simple returns: daily × periods_per_year.

    Arithmetic (not compounded) annualization is used so that linear
    relationships between volatility and return are preserved, e.g. the capital
    allocation line stays a straight line and the Sharpe ratio scales by
    sqrt(periods_per_year). Units are preserved, so percent per day becomes
    percent per year. NaN values pass through unchanged.

    Parameters
    ----------
    daily_return : float or array-like of float
        Mean daily simple return(s).

    periods_per_year : int, optional
        Trading days per year. Defaults to 252.

    Returns
    -------
    np.ndarray or np.float64
        Annualized return(s): an array with the same shape as an array-like
        input, or a NumPy scalar for a scalar input.
    """
    return np.asarray(daily_return, dtype=float) * periods_per_year


def annualize_risk(
    daily_risk: float | np.ndarray | list[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray | np.float64:
    """
    Annualize volatility (the standard deviation of daily returns): daily σ * sqrt(periods_per_year).

    Assumes daily returns are serially uncorrelated. Units are preserved, so
    percent per day becomes percent per year. NaN values pass through
    unchanged.

    Parameters
    ----------
    daily_risk : float or array-like of float
        Volatility: standard deviation(s) of daily returns.

    periods_per_year : int, optional
        Trading days per year. Defaults to 252.

    Returns
    -------
    np.ndarray or np.float64
        Annualized standard deviation(s): an array with the same shape as an
        array-like input, or a NumPy scalar for a scalar input.
    """
    return np.asarray(daily_risk, dtype=float) * np.sqrt(periods_per_year)


def _budget_constraint(weights: np.ndarray) -> float:
    """Fully invested: zero when the weights sum to 1."""
    return float(np.sum(weights) - 1.0)


@dataclass(frozen=True)
class PortfolioStats:
    """
    Annualized statistics of one portfolio (floats) or of several (arrays
    with one entry per portfolio), as returned by
    PortfolioProblem.portfolio_stats(). A portfolio whose weights contain NaN
    (e.g. an unreached efficient-frontier target) has NaN statistics.

    Attributes
    ----------
    annualized_return : float or np.ndarray
        Expected return, annualized arithmetically, in percent per year.

    risk : float or np.ndarray
        Volatility (standard deviation of returns), annualized, in percent per
        year. Named risk for compatibility with the uploaded keys.

    sharpe_ratio : float or np.ndarray
        Annualized Sharpe ratio (dimensionless): (annualized_return − annual
        risk-free rate) / risk.
    """

    annualized_return: float | np.ndarray
    risk: float | np.ndarray
    sharpe_ratio: float | np.ndarray


@dataclass(frozen=True, eq=False)
class PortfolioProblem:
    """
    One mean-variance portfolio problem: everything the optimizers, the
    random portfolios, and the uploaded statistics need. Immutable: the
    arrays are copied and made read-only on construction.

    Attributes
    ----------
    tickers : tuple[str, ...]
        Assets, in the order of the arrays below and of every weight vector.

    daily_mean_returns : np.ndarray
        Mean daily simple return of each asset, in percent, shape (n,).

    daily_cov : np.ndarray
        Covariance matrix of daily simple returns, in percent², shape (n, n).

    daily_risk_free_rate : float
        Daily risk-free rate in percent (annual rate in percent / 252).

    weight_cap_multiple : float
        Each weight is at most weight_cap_multiple × the equal weight, i.e.
        weight_cap_multiple / n. At least 1, or no portfolio is fully
        invested; above n the cap never binds.

    periods_per_year : int, optional
        Trading days per year for annualization. Defaults to 252.
    """

    tickers: tuple[str, ...]
    daily_mean_returns: np.ndarray
    daily_cov: np.ndarray
    daily_risk_free_rate: float
    weight_cap_multiple: float
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        tickers = tuple(self.tickers)
        n = len(tickers)
        if n == 0:
            raise ValueError("A portfolio problem needs at least one ticker.")
        mean = np.array(self.daily_mean_returns, dtype=float)
        cov = np.array(self.daily_cov, dtype=float)
        if mean.shape != (n,) or cov.shape != (n, n):
            raise ValueError(
                f"Expected mean returns of shape ({n},) and a covariance of shape "
                f"({n}, {n}) for {n} tickers, got {mean.shape} and {cov.shape}."
            )
        if not (np.isfinite(mean).all() and np.isfinite(cov).all()):
            raise ValueError("Mean returns and covariance must not contain NaN or inf.")
        if not np.isfinite(self.daily_risk_free_rate):
            raise ValueError("The daily risk-free rate must be finite.")
        if not self.weight_cap_multiple >= 1:
            raise ValueError(
                "weight_cap_multiple must be at least 1, or no portfolio can be "
                f"fully invested; got {self.weight_cap_multiple}."
            )
        mean.setflags(write=False)
        cov.setflags(write=False)
        object.__setattr__(self, "tickers", tickers)
        object.__setattr__(self, "daily_mean_returns", mean)
        object.__setattr__(self, "daily_cov", cov)
        object.__setattr__(self, "daily_risk_free_rate", float(self.daily_risk_free_rate))

    @classmethod
    def from_prices(
        cls,
        wide_df: pd.DataFrame,
        daily_risk_free_rate: float,
        weight_cap_multiple: float,
    ) -> "PortfolioProblem":
        """
        Estimate the problem from adjusted close prices: simple daily returns
        in percent, their mean, and their sample covariance (ddof = 1).

        Parameters
        ----------
        wide_df : pd.DataFrame
            Wide adjusted close prices (tickers in columns, sorted
            DatetimeIndex), as returned by data.prices.pivot_close_wide().

        daily_risk_free_rate : float
            Daily risk-free rate in percent.

        weight_cap_multiple : float
            See the class attributes.

        Returns
        -------
        PortfolioProblem
            The estimated problem.
        """
        returns_df = wide_df.pct_change().iloc[1:] * 100
        return cls(
            tickers=tuple(wide_df.columns),
            daily_mean_returns=returns_df.mean().to_numpy(),
            daily_cov=returns_df.cov().to_numpy(),
            daily_risk_free_rate=daily_risk_free_rate,
            weight_cap_multiple=weight_cap_multiple,
        )

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.tickers)

    @property
    def weight_cap(self) -> float:
        """Largest allowed weight of one asset: weight_cap_multiple / n."""
        return self.weight_cap_multiple / self.n_assets

    @property
    def annual_risk_free_rate(self) -> float:
        """The risk-free rate annualized like returns, in percent per year."""
        return float(annualize_return(self.daily_risk_free_rate, self.periods_per_year))

    def equal_weights(self) -> np.ndarray:
        """
        The equally weighted portfolio, 1 / n each. Always feasible (the cap
        is at least 1 / n), so the optimizers start from it.
        """
        return np.full(self.n_assets, 1.0 / self.n_assets)

    def bounds(self) -> list[tuple[float, float]]:
        """
        Bounds of each weight for scipy.optimize.minimize: long-only (≥ 0)
        and at most weight_cap.
        """
        return [(0.0, self.weight_cap)] * self.n_assets

    def constraints(self) -> list[dict]:
        """
        Constraints other than the bounds, for scipy.optimize.minimize: the
        weights sum to 1. A new list on every call, so a caller may append
        its own constraints (e.g. a target return).
        """
        return [{"type": "eq", "fun": _budget_constraint}]

    def single_asset_weights(self) -> np.ndarray:
        """
        One row per asset holding only that asset, shape (n, n). Pass to
        portfolio_stats() for each asset's own annualized return and volatility.
        These portfolios ignore the weight cap, so they are generally not
        feasible; they describe the assets, not candidate portfolios.
        """
        return np.eye(self.n_assets)

    def max_return_weights_closed_form(self) -> np.ndarray:
        """
        The highest-return portfolio under this problem's constraints, in
        closed form: fill the assets in order of decreasing mean return, each
        up to weight_cap, until the weights sum to 1 (with a cap of 40 %:
        40 % in the best asset, 40 % in the second, 20 % in the third).

        This is exact for this constraint set (long-only, fully invested,
        the same cap for every asset): moving weight from a filled asset to
        a later one can only lower the return. Ties in mean return make the
        weights non-unique but not the return. This is the maximum return
        portfolio that is uploaded, and it sets the top of the efficient
        frontier.

        Returns
        -------
        np.ndarray
            Weights as fractions, shape (n,), in the order of tickers.
        """
        weights = np.zeros(self.n_assets)
        remaining = 1.0
        for i in np.argsort(-self.daily_mean_returns, kind="stable"):
            weights[i] = min(self.weight_cap, remaining)
            remaining -= weights[i]
            if remaining <= 0:
                break
        return weights

    def max_constraint_violation(self, weights: np.ndarray) -> float | np.ndarray:
        """
        How far weights are from the constraint set: the largest of the
        negative part of any weight, any weight's excess over weight_cap,
        and |sum of weights − 1|. Zero for a feasible portfolio; NaN for a
        row containing NaN.

        Parameters
        ----------
        weights : np.ndarray
            One portfolio, shape (n,), or several, shape (m, n).

        Returns
        -------
        float or np.ndarray
            The violation of each portfolio: a float for one portfolio, an
            array of shape (m,) for several.
        """
        w = np.asarray(weights, dtype=float)
        violation = np.maximum.reduce(
            [
                (-w).max(axis=-1),
                (w - self.weight_cap).max(axis=-1),
                np.abs(w.sum(axis=-1) - 1.0),
            ]
        )
        return float(violation) if w.ndim == 1 else violation

    def daily_mean_return(self, weights: np.ndarray) -> float | np.ndarray:
        """
        Expected daily return w·μ in percent, of one portfolio (shape (n,),
        returns a float) or several (shape (m, n), returns shape (m,)).
        """
        mean = np.asarray(weights, dtype=float) @ self.daily_mean_returns
        return float(mean) if np.ndim(mean) == 0 else mean

    def daily_variance(self, weights: np.ndarray) -> float | np.ndarray:
        """
        Variance of daily returns wᵀΣw in percent², of one portfolio (shape
        (n,), returns a float) or several (shape (m, n), returns shape (m,)).
        Tiny negative values from rounding are clipped to 0; NaN passes
        through.
        """
        w = np.asarray(weights, dtype=float)
        variance = np.maximum(np.einsum("...i,ij,...j->...", w, self.daily_cov, w), 0.0)
        return float(variance) if np.ndim(variance) == 0 else variance

    def portfolio_stats(self, weights: np.ndarray) -> PortfolioStats:
        """
        Annualized return, volatility, and Sharpe ratio of one or several
        portfolios. Every portfolio statistic that is uploaded is computed
        here.

        Parameters
        ----------
        weights : np.ndarray
            One portfolio, shape (n,), or several, shape (m, n), as fractions
            (summing to 1), in the order of tickers.

        Returns
        -------
        PortfolioStats
            Floats for one portfolio, arrays of shape (m,) for several.
        """
        annualized_return = annualize_return(
            self.daily_mean_return(weights), self.periods_per_year
        )
        risk = annualize_risk(
            np.sqrt(self.daily_variance(weights)), self.periods_per_year
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_ratio = (annualized_return - self.annual_risk_free_rate) / risk
        if np.ndim(weights) == 1:
            return PortfolioStats(
                float(annualized_return), float(risk), float(sharpe_ratio)
            )
        return PortfolioStats(annualized_return, risk, sharpe_ratio)

    def sample_feasible_weights(
        self, n_portfolios: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Draw random portfolios uniformly from the constraint set: Dirichlet(1)
        draws (uniform on the long-only, fully invested simplex), keeping
        only those whose every weight is at most weight_cap (rejection
        sampling). The result is uniform on the capped simplex. Uniform draws
        rarely land near its corners, where the extreme portfolios are, so
        the cloud does not fill the feasible region.

        Parameters
        ----------
        n_portfolios : int
            Number of portfolios, at least 1.

        rng : np.random.Generator
            Seeded generator, so the draws are reproducible.

        Returns
        -------
        np.ndarray
            Weights of shape (n_portfolios, n), each row feasible.

        Raises
        ------
        ValueError
            If n_portfolios < 1, or if fewer than n_portfolios draws were
            accepted after MAX_SAMPLER_DRAWS_PER_PORTFOLIO × n_portfolios
            draws (the cap leaves almost none of the simplex).
        """
        if n_portfolios < 1:
            raise ValueError(f"n_portfolios must be at least 1, got {n_portfolios}.")
        alpha = np.ones(self.n_assets)
        accepted: list[np.ndarray] = []
        n_accepted = 0
        n_drawn = 0
        max_draws = MAX_SAMPLER_DRAWS_PER_PORTFOLIO * n_portfolios
        while n_accepted < n_portfolios:
            if n_drawn >= max_draws:
                raise ValueError(
                    f"Only {n_accepted} of {n_drawn} random portfolios satisfied the "
                    f"weight cap of {self.weight_cap:.4f}; raise weight_cap_multiple."
                )
            draws = rng.dirichlet(alpha, size=n_portfolios)
            n_drawn += n_portfolios
            feasible = draws[(draws <= self.weight_cap).all(axis=1)]
            accepted.append(feasible)
            n_accepted += len(feasible)
        return np.concatenate(accepted)[:n_portfolios]
