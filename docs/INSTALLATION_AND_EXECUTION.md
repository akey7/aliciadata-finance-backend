# Installation

## Package Management

This project uses [uv](https://docs.astral.sh/uv/) for Python versions, virtual environments, and dependencies. Install `uv` first (for example, `brew install uv` on macOS). Dependencies are declared in `pyproject.toml`, and `uv.lock` pins the exact versions that were resolved. The project requires Python 3.13 and pins 3.13.5 in `.python-version`, which `uv` downloads automatically if it is not already installed.

### Development Dependencies

Create the `.venv` virtual environment and install production plus development dependencies (the `dev` group is included by default). This also installs the `aliciadata_finance_backend` package from `src/` in editable mode, so edits to the library code take effect immediately:

```
uv sync
```

### Production Dependencies

To install only the production dependencies:

```
uv sync --no-dev
```

### Running the backend

One command runs everything: it downloads prices (or reuses today's cached download), fits the walk-forward ARIMA forecasts (or reuses the cached forecasts for the data's as-of date), optimizes the portfolio, backtests it against equal weights, and uploads all results for the frontend. Run it through `uv` so it uses the project environment; there is no need to activate the virtual environment manually:

```
uv run python scripts/run_optimization_and_time_series.py
```

Options:

| Flag | Effect |
| --- | --- |
| `--no-upload` | Write the `output/` files but upload nothing. Use this to check a run without touching any bucket. |
| `--retrain-arima` | Retrain the ARIMA models even if forecasts for the as-of date are cached. |
| `--refresh-data` | Download prices again even if today's download is cached. |
| `--workers N` | Number of worker processes for ARIMA training. Defaults to half the logical CPUs (for example 8 on a 16-thread Intel CPU, 5 on a 10-core M4). `--workers 1` trains everything in one process. The forecasts do not depend on this setting. |
| `--config PATH` | Model settings file to use instead of `config/modeling_and_optimization.yml` (see below). |

Nothing is uploaded unless every step succeeds. If a later step fails, the ARIMA forecasts already written to `output/` are reused on the next run.

Always run scripts from the repository root, because the scripts read and write `input/` and `output/` (and read `config/`) relative to the current working directory.

### Model settings

Time series, portfolio optimization, and portfolio backtest settings live in `config/modeling_and_optimization.yml`, which is checked into the repository. **For the current values, and what each setting means, consult that file's comments;** the validation rules (types and minimums) are documented in `src/aliciadata_finance_backend/config.py`. This document deliberately does not repeat the values, so it cannot go out of date when they change.

```yaml
time_series_models:
  training_window_length: ...     # closes (trading sessions) per ARIMA training window
  evaluation_window_length: ...   # most recent trading days backtested and scored
portfolio_optimization:
  weight_cap_multiple: ...        # each weight ≤ this / number of tickers
  efficient_frontier_points: ...  # target returns on the efficient frontier
  simulated_portfolios: ...       # random feasible portfolios in the plot data
  random_seed: ...                # seed for those random portfolios
  near_frontier_tolerance: ...    # pp of return counted as near the frontier (metadata only)
portfolio_backtest:
  training_window_returns: ...     # daily returns per optimization, live and backtest
  rebalance_interval_sessions: ... # trading sessions between rebalances
```

- `training_window_length` counts **closing prices**; each window has one fewer daily log return than closes.
- `evaluation_window_length` is the number of backtest days in `all_forecasts.csv` (plus the live forecast) and in the metrics of `all_forecasts_metadata.yml`.
- Every key is required, and unknown keys are rejected, so a typo stops the run before any download instead of silently using a default.
- The run downloads enough history for the ARIMA windows plus scored days and for the backtest's training window plus scored days, whichever is longer, with a small margin.
- Each run records the time series values it used in `all_forecasts_metadata.yml` (`forecast_window.training_window_returns`, `evaluation_window.target_trading_days`), which is where the frontend reads them.
- The portfolio settings apply to one shared problem definition: the weight cap constrains the max Sharpe ratio and minimum variance portfolios, the efficient frontier, and the random portfolios alike. `efficient_frontier_points` and `simulated_portfolios` set the lengths of those arrays in `portfolio_optimization_plot_data.h5`, which the frontend reads from the file. The same seed and data draw the same random portfolios.
- The portfolio backtest scores the same `evaluation_window_length` days as the forecasts and uses the same `weight_cap_multiple` as the optimization. `training_window_returns` counts **daily returns**, so each window holds one more close than this, and it also sets the live optimization's window, so changing it changes the optimization files too. `rebalance_interval_sessions` must divide `evaluation_window_length` exactly, or the run stops before any download. Each run records these values in `portfolio_backtest_metadata.yml`.

Changing a value changes what is uploaded. After changing a time series setting, run with `--retrain-arima`, because the cached forecasts for a date are otherwise reused; the optimization and the backtest are recomputed on every run. `INTERFACE_CONTRACT.md` does not depend on the values, so changing them needs no contract update as long as their meaning stays the same.

### Running the tests

The tests in `tests/` run in memory: they use mocks and temporary directories, and never touch the network, the bucket, `input/`, or `output/`. Run them from the repository root:

```
uv run pytest
```

Run other development tools the same way, for example `uv run black .`.

### Managing dependencies

```
uv add <package>           # add a production dependency
uv add --dev <package>     # add a development dependency
uv remove <package>        # remove a dependency
uv lock --upgrade          # upgrade all packages within the declared constraints
uv sync                    # bring .venv in line with uv.lock
```

Commit both `pyproject.toml` and `uv.lock` whenever dependencies change.

### `.env` file

**Never commit the `.env` file to GitHub!** It should never be made publicly accessible because it contains API keys.

Here are the keys the scripts need to function:

1. `MASSIVE_API_KEY`: Key to Massive.com API for stock and index data. Obtain from API provider.

2. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL_S3`: key id, secret key, region (e.g. `nyc3`), and endpoint of the DigitalOcean Space. These are boto3's standard names. The key needs read/write/delete permission on the Space. `AWS_ENDPOINT_URL_S3` must be the region-only endpoint (e.g. `https://nyc3.digitaloceanspaces.com`) without the Space name; the uploader refuses to run otherwise. Obtain the values from DigitalOcean.

3. `S3_BUCKET_NAME`: Name of the one DigitalOcean Space (S3 bucket) that holds all frontend data.

4. `S3_PORTFOLIO_OPTIMIZATION_FOLDER` and `S3_TIME_SERIES_FOLDER`: names of the two top-level folders inside that Space, holding the portfolio optimization and ARIMA time series data. Dev and production share the Space and use different folder names. Files are uploaded to `<folder>/YYYY-MM-DD/<file>`; see `INTERFACE_CONTRACT.md` for the full key layout.

Items 2 through 4 connect the backend to the front end. An upload run fails before any work starts if any of them is missing or empty. The scripts load `.env` with its values taking precedence over variables of the same name already set in the shell, so AWS credentials exported for other work are never sent to DigitalOcean. These names replaced the `FSF_FRONT_END_*` names (contract v45); the old names are no longer read.

## Execution

Once setup is complete, run the whole pipeline from the repository root as described in [Running the backend](#running-the-backend).
