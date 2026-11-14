# Interface Contract: aliciadata-finance-backend (backend) ↔ aliciadata-finance-frontend (frontend)

| | |
| --- | --- |
| **Contract version** | 48 |
| **Last updated** | 2026-11-14 |
| **Source of truth** | `aliciadata-finance-backend` (backend, formerly `fuzzy-system-research`). The copy in `aliciadata-finance-frontend` (formerly `fuzzy-system-finance`) is a mirror. |

This file is stored **byte-for-byte identically** in both repos. It describes the files the backend uploads to one S3-compatible bucket (a DigitalOcean Space, accessed with `boto3`) and the frontend reads. Treat it as a public API.

## Rules for editing this file (humans and AI agents)

1. **Edit the backend copy first**, then copy it unchanged into the frontend repo. Never edit only one copy, and never let the copies diverge.
2. **Bump the contract version and add a changelog entry** for every change, including clarifications that change meaning.
3. **Breaking changes** (anything in the list below) need explicit human approval. Describe the frontend impact before making the change, and prefer an additive alternative with a migration path.
   - Renaming or removing an object key, top-level folder, file, YAML key, HDF5 group/dataset, or CSV column
   - Changing a data type, units, scaling, or date format
   - Changing the shape of an array or the meaning of a value
   - Changing the set of tickers or the covered date range in a way a reader could not detect from the data itself
4. **Additive changes** (new optional keys, new columns, new files) are allowed without breaking the contract, **because readers must ignore anything they do not recognize** (see "Reader requirements").
5. The backend agent must not edit the frontend repo, and the frontend agent must not edit the backend repo. Each lists what the other must change.

## Status of this document

Version 1 was written from the backend code. Version 2 was checked against real output files from the backend (a real `optimization_metadata.yml`, the `h5dump -H` structure of the `.h5` file, and the header and final row of `all_forecasts.csv`). What was and was not verified is listed under "Verification status" at the end.

## Where the data lives

All data lives in **one DigitalOcean Space**, which is one S3 bucket. Inside it are **two top-level folders** (key prefixes), whose names are the values of two environment variables. Both the backend and the frontend use these variable names:

| Environment variable | Meaning |
| --- | --- |
| `S3_BUCKET_NAME` | Name of the Space (the S3 bucket) |
| `S3_PORTFOLIO_OPTIMIZATION_FOLDER` | Name of the top-level folder holding the optimization files |
| `S3_TIME_SERIES_FOLDER` | Name of the top-level folder holding the ARIMA forecasts |

**Dev and production share the one Space and differ only in the folder names.** Pointing a frontend or backend at dev or production means setting these two folder variables, so both sides must use the same pair of values for the same environment. Since contract v45 the three variables are named as in the table. Before v45 they were `FSF_FRONT_END_BUCKET_NAME`, `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`, and `FSF_FRONT_END_TIME_SERIES_FOLDER`, and before v13 the folder variables were `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME`. The backend reads only the current names.

Every file is stored under a **date prefix** `YYYY-MM-DD/` inside its folder, one date folder per run, so past runs are kept. Full object keys within the Space, where `<X>` means the value of environment variable `X`:

```
<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml
<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_optimization_plot_data.h5
<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_backtest_daily.csv      (since v36)
<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_backtest_metadata.yml   (since v36)
<S3_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv
<S3_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts_metadata.yml   (since v21)
```

Dev and production are selected by the configured Space and folder names; the key patterns are identical in both. In this document, **"folder"** means one of these two top-level folders, and **"date folder"** means a `YYYY-MM-DD/` prefix inside one.

### Connecting

- **Endpoint:** use the **region-only** endpoint `https://<region>.digitaloceanspaces.com`, pass the **Space name as the S3 `Bucket`**, and pass the full key (starting with the folder name) as the S3 `Key` or `Prefix`.
- **Space name variable:** both the backend and the frontend read the Space name from `S3_BUCKET_NAME`. The backend uploads with `Bucket=<space>` and `Key="<folder>/YYYY-MM-DD/<file>"`, and refuses to run with an endpoint that names the Space.
- **Credentials, region, and endpoint** are each side's own configuration, not part of this contract. Since v45 the backend uses boto3's standard names (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL_S3`); the frontend may use the same names for its own key, but is not required to.
- **Avoid the Space-specific endpoint:** do **not** put the Space name in the endpoint host (`https://<space>.<region>.digitaloceanspaces.com`) and pass a folder name as `Bucket`. Uploads happen to land at the right key that way (boto3 sends path-style requests, so the "bucket" becomes the first key segment), but listings fail with `NoSuchKey`.

### The date prefix

- **Meaning:** `YYYY-MM-DD` is the **data as-of date**: the last NYSE trading day in the price data used by that run, not the day the backend ran. A run on a Sunday stores Friday's results under Friday's date. Since v47, the backend can also be run for a **past** session: that run uses only prices up to that date, as if the data ended then, so its files are those a run on that day's data would give (see "Re-runs and past dates").
  - For the optimization files it equals `date_updated.date_to` in that date folder's `optimization_metadata.yml`.
  - For the backtest files it equals `evaluation_window.end_date` in `portfolio_backtest_metadata.yml`.
  - For `all_forecasts.csv` it is the last `pred_date` with a non-empty actual price. The final row (the live forecast) is the next trading day after the folder date.
- **Re-runs and past dates** (since v47). A run writes only its own date folder, replacing the files already in it, and never modifies or deletes any other date folder. Most runs are for the latest session; re-running the same date overwrites that date's files. A run for a past session (the backend's as-of mode) writes, or overwrites, that past date's folder, so a date folder can be computed after its date, with the method and settings current at that time and the adjusted prices as Massive.com returned them then, which can differ from those available on the date itself (for example after a later split or distribution). Before v47 the backend wrote only the latest date. Readers must not assume a date folder was computed on its date or never changes. Since v48, every metadata file (`optimization_metadata.yml`, `portfolio_backtest_metadata.yml`, and `all_forecasts_metadata.yml`) records when its date folder was computed (`computed_at`, UTC) and the contract version it was written under (`contract_version`), so a folder computed after its date can be recognized: its `computed_at` is later than its date. A one-time backfill of many past dates writes each date this way, oldest first.
- **Format:** the date prefix is the **first path segment after the top-level folder**. After removing `<folder>/` from the start of a key, the rest must start with exactly `^\d{4}-\d{2}-\d{2}/`, zero-padded. Lexicographic order equals chronological order.
- **Stale objects:** readers must **ignore every key whose segment after the folder is not a valid date**. In particular, objects left directly under a folder by contract v8 and earlier (for example `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/optimization_metadata.yml`, `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/portfolio_optimization_plot_data.h5`, `<S3_TIME_SERIES_FOLDER>/all_forecasts.csv`) are stale and are no longer written; the human removes them by hand.
- **Retention:** all dates are kept indefinitely. There is no pruning.

### Which dates are available

A date is **available** only when its data is complete in **both** folders:

1. Key `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml` exists. The backend uploads this file last of all the files for a date, so its presence means both optimization files for that date are complete (and, since v10, that the forecasts were uploaded just before).
2. Key `<S3_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv` exists.

Since v10, one backend run computes everything first and then uploads its files in the order `all_forecasts.csv`, `all_forecasts_metadata.yml` (since v21), `portfolio_optimization_plot_data.h5`, `portfolio_backtest_daily.csv` and `portfolio_backtest_metadata.yml` (since v36), `optimization_metadata.yml`; if any computation fails, nothing is uploaded. Because `optimization_metadata.yml` is last, its presence (for v21+ runs) also means the forecasts metadata is complete, and for v36+ runs that both backtest files are. `all_forecasts_metadata.yml` and the backtest files are not part of the availability rule, since older dates do not have them. Readers must still check both folders, because dates written before v10 came from two separate scripts. Dates present in only one folder (for example, only one of the two pre-v10 scripts was run) are **not** available and must not be offered to users.

**To find the dates,** list each folder with S3 `ListObjectsV2` using `Bucket=<space>` and `Prefix="<folder>/"`:

- Follow `ContinuationToken` pagination, since a listing returns at most 1,000 entries.
- Strip the `<folder>/` prefix from each returned key, then match the date pattern above.
- `Delimiter="/"` is optional. With it, date folders come back as `CommonPrefixes` such as `<folder>/2026-10-09/`, and you must still strip the folder prefix, then confirm the marker object (a `HeadObject`, or a listing with `Prefix="<folder>/YYYY-MM-DD/"`). Without it, the full keys are returned and the marker objects can be matched directly.

General behavior of every file:

- **History.** Each date folder is a snapshot of one run. Comparing dates is valid (same keys, formats, and units within a contract version), but a date folder written under an older contract version keeps the format of that version.
- **Non-atomic.** The files are uploaded one after the other (forecasts, forecasts metadata, `.h5`, the backtest files, then optimization metadata). The completion marker rule above prevents reading a half-uploaded date. While a date is re-run, its files can briefly come from different runs.
- **Manual cadence.** The backend is run by hand, so there is no guaranteed update schedule. The frontend should display the data dates (see below) so a stale dataset is visible, rather than assuming data is current.
- **Content type and ACLs.** The backend sets none. Do not rely on the `Content-Type` of an object; decide how to parse from the key name.
- **Not investment advice: one disclaimer, shown once.** Display this wording exactly once, at the bottom of the UI:

  > Not investment advice. Every value shown is a backtest or model output computed from historical prices, and none of it predicts future results or recommends any investment.

  It covers every file in this contract. Do not repeat it in plot captions, tooltips, or metric descriptions; the suggested wording elsewhere in this document deliberately leaves it out.

## Shared conventions

- **Tickers** are the strings used by Massive.com, e.g. `NVDA`, `QQQ`. At the time of writing the set is `ETHA, FXE, GLD, IBIT, NVDA, QQQ, VOO, VXUS, XBI, XLE`, but **the frontend must not hardcode it**. Read the ticker list from `optimization_metadata.yml` (`tickers`) or from the `all_forecasts.csv` column names. The set can change.
- **Prices** are daily close prices in US dollars as returned by Massive.com with `adjusted=true`, i.e. adjusted by Massive.com's rules. Treat them as "adjusted close" and do not assume dividend adjustment.
- **Returns** are computed from **simple** (not log) daily returns of consecutive adjusted closes and expressed in **percent** (1.5 means 1.5 %).
- **Volatility** means the standard deviation (σ) of returns, in percent. Fields and attributes named `risk` (for example `optimum_portfolio.risk` and the `.h5` attribute `risk_units`) hold volatility; they keep that name for compatibility.
- **Every return and volatility in the two optimization files is annualized.** (The backtest files have their own conventions, including daily returns and compounded growth; see "`portfolio_backtest_metadata.yml`".) The backend optimizes in daily units and annualizes only when writing files, using 252 trading days per year and **arithmetic** (not compounded) annualization:
  - annualized return = mean daily return × 252
  - annualized volatility = daily σ × √252
  - annualized Sharpe ratio = daily Sharpe ratio × √252, which equals (annualized return − annual risk-free rate) / annualized volatility

  Arithmetic annualization keeps volatility/return relationships linear, so the capital allocation line (the `tangency_line` group of the `.h5` file) is straight, its intercept is the quoted annual risk-free rate, and its slope is the annualized Sharpe ratio. It is not the compounded growth an investor would realize. If user-facing text needs that distinction, say "annualized (arithmetic)" rather than "annual growth".
- The **only** daily quantity in the optimization files is `risk_free_rate.daily_risk_free_rate`, kept for backward compatibility and named accordingly.
- **Dates** are NYSE trading days (sessions of the New York Stock Exchange calendar, so weekends and NYSE holidays such as Good Friday are excluded, while federal-only holidays such as Columbus Day are included). "Business day" in this document means NYSE trading day. Date-only fields are strings in `YYYY-MM-DD` format.

---

## `optimization_metadata.yml`

**Folder:** `S3_PORTFOLIO_OPTIMIZATION_FOLDER`. **Key:** `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml`. **Format:** YAML (block style, keys sorted alphabetically), UTF-8. Parse it with a safe YAML loader. Date-like values are written as **quoted strings**, so loaders return strings, not date objects.

### Example

A real file produced by the backend (contract v4+, data through 2026-10-09):

```yaml
annualization:
  method: arithmetic
  periods_per_year: 252
date_updated:
  date_from: '2025-10-07'
  date_to: '2026-10-09'
min_var_portfolio:
  annualized_return: 12.851676947885277
  risk: 6.89459351276483
optimum_portfolio:
  annualized_return: 34.343418898244956
  risk: 12.439838563729614
  sharpe_ratio: 2.4351938928347816
  weights:
    ETHA: 7.59701304408773e-15
    FXE: 3.055998668072217e-14
    GLD: 4.060464045601646e-14
    IBIT: 0.0
    NVDA: 0.0
    QQQ: 19.079171927481532
    VOO: 13.836265883217862
    VXUS: 0.0
    XBI: 27.084562189300787
    XLE: 39.99999999999994
risk_free_rate:
  daily_risk_free_rate: 0.01607142857142857
  risk_free_rate: 4.05
  risk_free_rate_date: '2026-10-08 00:00:00'
tickers:
- XBI
- QQQ
- XLE
- VOO
- NVDA
- VXUS
- GLD
- IBIT
- FXE
- ETHA
```

Things to notice in the example:

- Date strings are quoted, so they load as strings.
- `weights` keys are sorted alphabetically, but `tickers` is in the backend's own order. Neither order is significant.
- Weights carry **floating-point noise**: a weight that is effectively zero can appear as `7.59701304408773e-15`, and a weight at the cap appears as `39.99999999999994` instead of `40.0`. Values can be written in scientific notation.

Since v30, the file has one comment line directly above the `optimum_portfolio` key (the example above predates it), because the key's name does not say which portfolio it is:

```yaml
# optimum_portfolio is the maximum Sharpe ratio (tangency) portfolio.
optimum_portfolio:
```

YAML parsers ignore comments, so the parsed contents are unchanged. The key keeps its name for compatibility.

Since v28, `min_var_portfolio` also has a `weights` map, in the same form as `optimum_portfolio.weights` (the example above predates it). Illustrative values:

```yaml
min_var_portfolio:
  annualized_return: 12.851676947885277
  risk: 6.89459351276483
  weights:            # one key per ticker; only three shown here
    ETHA: 1.2e-15
    FXE: 39.99999999999998
    GLD: 21.403
```

Since v32, the file also has four sections that describe the assets, the constraints, the top of the efficient frontier, and the random portfolios in the `.h5` file (the example above predates them). Illustrative values; one key per ticker in `assets` and in `max_return_portfolio.weights`, only three shown:

```yaml
assets:
  ETHA:
    annualized_return: -21.7
    risk: 71.2
  GLD:
    annualized_return: 38.9
    risk: 19.4
  NVDA:
    annualized_return: 41.3
    risk: 48.0
constraints:
  fully_invested: true
  long_only: true
  weight_cap: 40.00000000000001
  weight_cap_multiple: 4
max_return_portfolio:
  annualized_return: 38.1
  risk: 15.4
  weights:
    ETHA: 0.0
    GLD: 40.0
    NVDA: 19.99999999999998
random_portfolios:
  count: 10000
  equal_weight_annualized_return: 11.2
  max_annualized_return: 32.2
  mean_annualized_return: 11.3
  near_frontier_share: 0.0004
  near_frontier_tolerance: 1.0
  random_seed: 20261025
  return_std: 5.9
  return_std_uniform_without_cap: 6.4
  sampling: uniform on the constraint set
```

### Fields

| Path | Type | Meaning |
| --- | --- | --- |
| `date_updated.date_from` | string `YYYY-MM-DD` | First trading day of price data used in the optimization. |
| `date_updated.date_to` | string `YYYY-MM-DD` | Last trading day of price data used. **This is the "data as of" date to show users.** |
| `tickers` | list of string | Tickers in the optimized portfolio. Order is not significant. |
| `risk_free_rate.risk_free_rate` | float | The last published 3-month U.S. Treasury bill rate (FRED series `DTB3`) dated **before** the date folder's date (since v38; before that, the most recent rate when the backend ran), **annualized, in percent** (4.39 means 4.39 % per year). Days with no published value (bond-market holidays) are skipped, so this is never `NaN`; since v14, a run with no valid rate fails instead of uploading. |
| `risk_free_rate.daily_risk_free_rate` | float | `risk_free_rate / 252`. **Percent per day**, the rate the backend used internally for its daily optimization. The only daily quantity in the optimization files; prefer `risk_free_rate` for display. |
| `risk_free_rate.risk_free_rate_date` | string | Observation date of the rate (the last day before the date folder's date with a published value), formatted as a timestamp string such as `2026-10-08 00:00:00`. Parse the first 10 characters as a date. |
| `optimum_portfolio.weights` | map ticker → float | Weights of the maximum Sharpe ratio portfolio, **in percent** (a weight of 22.4 means 22.4 %). Contains one key per ticker. |
| `optimum_portfolio.annualized_return` | float | Expected return of the max Sharpe ratio portfolio, **annualized arithmetically, in percent**: mean daily return (%) × 252. *Changed in v4; it was compounded before.* |
| `optimum_portfolio.risk` | float | Volatility of that portfolio (standard deviation of its returns), **annualized** (daily σ × √252), **in percent**. |
| `optimum_portfolio.sharpe_ratio` | float | Annualized Sharpe ratio of that portfolio (dimensionless): daily Sharpe × √252 = (`annualized_return` − `risk_free_rate.risk_free_rate`) / `risk`. *Added in v4.* |
| `min_var_portfolio.annualized_return` | float | Expected return of the minimum variance portfolio, **annualized arithmetically, in percent**. *Added in v4.* |
| `min_var_portfolio.risk` | float | Volatility of the minimum variance portfolio (standard deviation of its returns), **annualized, in percent**. *Added in v4.* |
| `min_var_portfolio.weights` | map ticker → float | Weights of the minimum variance portfolio, **in percent**, one key per ticker, in the same form and with the same floating-point noise as `optimum_portfolio.weights`. *Added in v28; absent in older files, so treat it as optional.* |
| `annualization.method` | string | Always `arithmetic` in v4. Describes how every return and volatility in both optimization files was annualized. *Added in v4.* |
| `annualization.periods_per_year` | int | Trading days per year used for annualization (`252`). *Added in v4.* |
| `max_return_portfolio.annualized_return` | float | The highest expected return any portfolio can have under the optimizer's constraints, **annualized arithmetically, in percent**: the top of the efficient frontier (`efficient_frontier/ys` in the `.h5` file ends at exactly this value). Since v39 it is the exact closed-form answer (fill the highest-return assets up to the weight cap, in order), not a numerical optimization. *Added in v32.* |
| `max_return_portfolio.risk` | float | Its annualized volatility (standard deviation of returns), in percent. *Added in v32.* |
| `max_return_portfolio.weights` | map ticker → float | Its weights in percent, one key per ticker, with the same floating-point noise as the other weight maps. With a binding cap, the highest-return tickers sit at the cap. *Added in v32.* |
| `assets.<TICKER>.annualized_return`, `.risk` | float | Each ticker held **on its own**: mean daily return × 252 and daily σ × √252, in percent, from the same data as the optimization. With a binding cap these single-ticker portfolios are not feasible, so do not plot them as candidate portfolios; they explain where the cloud and the frontier sit. Ticker keys are uppercase, as in the weight maps. *Added in v32.* |
| `constraints.long_only`, `.fully_invested` | bool | Always `true`: weights are at least 0 and sum to 100 %. *Added in v32.* |
| `constraints.weight_cap` | float | Largest allowed weight of one ticker, **in percent** (`weight_cap_multiple` / number of tickers × 100). A value of 100 or more means the cap does not bind. *Added in v32.* |
| `constraints.weight_cap_multiple` | int or float | The backend setting the cap comes from. *Added in v32.* |
| `random_portfolios.count`, `.random_seed` | int | Number of random portfolios in `simulated_portfolios` and the seed they were drawn with. *Added in v32.* |
| `random_portfolios.sampling` | string | `uniform on the constraint set`. *Added in v32.* |
| `random_portfolios.max_annualized_return` | float | Highest return among the random portfolios, in percent. Compare with `max_return_portfolio.annualized_return`. *Added in v32.* |
| `random_portfolios.mean_annualized_return`, `.equal_weight_annualized_return` | float | The random portfolios' average return, and the equally weighted portfolio's, in percent. They agree up to sampling error, because uniform draws average to equal weights: the cloud is centered on the equal-weight portfolio. *Added in v32.* |
| `random_portfolios.return_std` | float | Standard deviation of the random portfolios' returns, in percentage points. *Added in v32.* |
| `random_portfolios.return_std_uniform_without_cap` | float | Its theoretical value for uniform draws without a cap: sqrt(Σ(Rᵢ − R̄)² / (n (n + 1))), where Rᵢ are the `assets` returns and n the number of tickers. The cloud's spread is roughly the tickers' own spread divided by sqrt(n + 1); a binding cap narrows it further, so `return_std` is normally smaller. *Added in v32.* |
| `random_portfolios.near_frontier_tolerance` | float | Percentage points of return (a backend setting). *Added in v32.* |
| `random_portfolios.near_frontier_share` | float | Fraction (0 to 1) of random portfolios whose return is within `near_frontier_tolerance` of the efficient frontier's return at the same volatility (straight lines between frontier points, which can only overstate the share). *Added in v32.* |
| `computed_at` | string | When the backend computed this date folder, in UTC, as `YYYY-MM-DDTHH:MM:SSZ` (for example `2026-11-14T12:00:00Z`): the start of the run that wrote it. All metadata files from one run have the same value. It can be long after the folder's date (see "Re-runs and past dates"). *Added in v48; absent in older files, so treat it as optional.* |
| `contract_version` | int | The version of this contract the date folder was written under (`48` or later). Files without it were written under an earlier version. *Added in v48; absent in older files, so treat it as optional.* |

### Notes

- **Weight noise.** These rules apply to every weight map (`optimum_portfolio.weights`, `min_var_portfolio.weights` since v28, and `max_return_portfolio.weights` since v32). Treat any weight with absolute value below `1e-6` (percent) as zero, and round for display. Never compare a weight to an exact value such as `0` or `40`, and never assume a "zero" weight is written as exactly `0.0`. Do not assume the weights sum to exactly 100; they sum to 100 only within the optimizer's tolerance.
- **Converged optimizations only** (since v29). If the max Sharpe ratio or minimum variance optimization does not converge, the backend run fails and nothing is uploaded for that date, so every weight, return, volatility, and Sharpe ratio in the optimization files comes from a converged optimization. (An efficient-frontier target return the optimizer cannot reach is still recorded as `NaN` in the `.h5` file; see that section. The backtest's past optimizations have a fallback instead; see "`portfolio_backtest_metadata.yml`".)
- Both portfolios are optimized under the same constraints: long-only, weights sum to 100 (within optimizer tolerance), and each weight is capped at a backend setting, currently `4 / (number of tickers)` of the portfolio (40 % with 10 tickers). Since v31 the efficient frontier and the random portfolios in the `.h5` file use exactly the same constraints. Those constraints are the optimizer's current settings, not a contractual guarantee. Do not hard-code them in the frontend.
- Inputs (mean returns and covariance) are estimated from roughly one year of daily data, so the optimum is noisy and sensitive to the window. Say so in user-facing text.
- **Weights of both portfolios.** Since v28 this file has the weights of both the max Sharpe ratio portfolio (`optimum_portfolio.weights`) and the minimum variance portfolio (`min_var_portfolio.weights`). Before v28, only the max Sharpe ratio weights were included; for those dates, show the minimum variance portfolio's return and volatility without weights. The minimum variance portfolio depends only on the estimated covariance of returns, not on expected returns or the risk-free rate, so its weights are usually more stable from one date to the next than the max Sharpe ratio weights.
- **Suggested wording near a table of both portfolios' weights:** "The minimum variance portfolio uses only the estimated covariance of returns, not expected returns, so it is usually more stable from one date to the next than the max Sharpe portfolio. Both are estimated from about a year of daily data and share the same constraints (long-only, capped weights)."
- `optimum_portfolio` and `min_var_portfolio` values equal the `max_sharpe_ratio` and `min_var_portfolio` points in the `.h5` file from the same run (same units). Since v38, `optimum_portfolio` also equals `final_optimization` in `portfolio_backtest_metadata.yml`.
- **Training window** (since v38). The optimization uses the same number of returns as each backtest optimization: `schedule.training_window_returns` in `portfolio_backtest_metadata.yml`. The window starts on `date_updated.date_from`. Before v38 it held one return fewer.
- **Why few random portfolios come near the efficient frontier** (since v32, the numbers are in the file). The random portfolios are drawn uniformly from the constraint set, so they average to equal weights: their mean return is the equal-weight portfolio's, and their spread is only about the tickers' own spread divided by sqrt(n + 1) (`random_portfolios.return_std` versus `return_std_uniform_without_cap`). The frontier's upper end needs most of the portfolio in a few tickers (`max_return_portfolio.weights`), a corner of the constraint set; with n tickers the share of uniform draws near a corner shrinks roughly like a power n − 1 of the distance, so with 10 tickers essentially none land there. `assets` shows which tickers pull the cloud toward high volatility and low return.
- **Suggested wording near the plot** (fill in `{share}` as `random_portfolios.near_frontier_share` × 100 to one decimal, `{count}` and `{tol}` from `random_portfolios`; leave the sentence out for dates before v32): "Only {share}% of the {count} random portfolios lie within {tol} percentage points of the efficient frontier. Uniform draws cluster around the equal-weight portfolio and rarely reach the corners of the constraint set, where the frontier's high-return portfolios are."
- **Detecting older files.** A file without an `annualization` key predates v4; its `annualized_return` is compounded and it has no Sharpe ratio or min-variance section. A file whose `min_var_portfolio` has no `weights` key predates v28. A file without `max_return_portfolio`, `assets`, `constraints`, and `random_portfolios` predates v32; treat all four as optional.

---

## `portfolio_optimization_plot_data.h5`

**Folder:** `S3_PORTFOLIO_OPTIMIZATION_FOLDER`. **Key:** `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_optimization_plot_data.h5`. **Format:** HDF5 (written with `h5py`). The file has five top-level groups. Each group has exactly two datasets named `xs` and `ys`: one-dimensional float64 arrays of equal length within the group. `xs` is the horizontal axis (volatility), and `ys` is the vertical axis (return). All values are **annualized (arithmetic), in percent** (see "Shared conventions"), and the file contains no dates and no ticker names (use `optimization_metadata.yml` for those).

### Root attributes (added in v4)

The root group `/` carries attributes that describe the units, so a reader can verify them instead of assuming:

| Attribute | Type | Value in v4 |
| --- | --- | --- |
| `return_units` | string | `percent_per_year` |
| `risk_units` | string | `percent_per_year` (annualized volatility: standard deviation of returns, in percent) |
| `return_type` | string | `simple` |
| `annualization` | string | `arithmetic` |
| `periods_per_year` | int64 | `252` |

**Detecting older files.** A file with **no** `annualization` attribute predates v4 and contains **daily** values in percent. Readers should check this attribute and refuse to plot (or convert explicitly) rather than mixing daily and annualized data.

### Layout

```
/                             attrs: return_units, risk_units, return_type, annualization, periods_per_year
/efficient_frontier/xs        float64, shape (100,)    annualized volatility σ (%) of the minimum-variance portfolio for each target return
/efficient_frontier/ys        float64, shape (100,)    the 100 target annualized returns (%), evenly spaced from the minimum variance portfolio's return to the highest feasible return
/tangency_line/xs             float64, shape (100,)    annualized volatility σ (%) on the capital allocation line, evenly spaced from 0 to exactly max_sharpe_ratio/xs
/tangency_line/ys             float64, shape (100,)    annualized return (%): risk_free_rate + sharpe_ratio × xs, ending exactly at max_sharpe_ratio/ys
/simulated_portfolios/xs      float64, shape (10000,)  annualized volatility σ (%) of random portfolios drawn from the optimizer's constraint set
/simulated_portfolios/ys      float64, shape (10000,)  annualized return (%) of those portfolios
/max_sharpe_ratio/xs          float64, shape (1,)      annualized volatility σ (%) of the max Sharpe ratio portfolio
/max_sharpe_ratio/ys          float64, shape (1,)      its annualized return (%)
/min_var_portfolio/xs         float64, shape (1,)      annualized volatility σ (%) of the minimum variance portfolio
/min_var_portfolio/ys         float64, shape (1,)      its annualized return (%)
```

In the capital allocation line formula, `risk_free_rate` and `sharpe_ratio` are the YAML fields `risk_free_rate.risk_free_rate` and `optimum_portfolio.sharpe_ratio` from the same run.

**One definition behind every point** (since v31). The optimizer's constraint set (long-only, weights sum to 100 %, and the per-ticker weight cap, a backend setting) applies to all five groups: the max Sharpe ratio and minimum variance portfolios, every efficient-frontier point, and every random portfolio. Every volatility and return in this file is computed by the same backend function, with the same annualization, as the values in `optimization_metadata.yml`. The backend checks before uploading that no random portfolio has a higher Sharpe ratio than the max Sharpe ratio point or lies left of the efficient frontier, that the minimum variance volatility equals the frontier's smallest volatility, and that the capital allocation line ends exactly at the max Sharpe ratio point; if any check fails, nothing is uploaded for that date.

### Group meanings

| Group | What to draw |
| --- | --- |
| `efficient_frontier` | A line. Since v31, points run from the minimum variance portfolio's return up to the **highest return that is feasible** under the optimizer's constraints (before v31: up to the highest simulated return). |
| `tangency_line` | The **capital allocation line (CAL)**: a straight line from (0, annual risk-free rate) to the max Sharpe ratio portfolio, with slope equal to the annualized max Sharpe ratio. Since v31 it **stops at the max Sharpe ratio point**, because portfolios beyond it would need borrowing (leverage), which is out of scope. The group keeps its name `tangency_line` for compatibility. The name "capital market line" used before v31 is retired: call it the capital allocation line. |
| `simulated_portfolios` | A scatter cloud of random portfolios, drawn uniformly from the same constraint set as the optimizer (since v31), to illustrate the feasible region. Illustrative only (see Notes). |
| `max_sharpe_ratio` | A single marker: the max Sharpe ratio (tangency) portfolio. |
| `min_var_portfolio` | A single marker: the minimum variance portfolio. |

### Notes

- **Same units as the YAML file.** Since v4, this file and `optimization_metadata.yml` are both annualized (arithmetic) in percent, so YAML values can be plotted directly on the same axes.
- **`NaN` values.** An `efficient_frontier/xs` entry is `NaN` when the optimizer did not reach that target return. Readers must drop or break the line at `NaN` and must not treat it as an error. Since v31 every target is feasible, so `NaN` is rare; it can still occur at the very top of the frontier, where the highest feasible return is reached by a single corner portfolio that the optimizer may not converge to. `efficient_frontier/ys` never contains `NaN`. `tangency_line` never contains `NaN` (since v5), and since v31 it no longer depends on the efficient frontier at all.
- **Older date folders: clip the line.** In date folders written before v31, `tangency_line` extends past the max Sharpe ratio point, to the largest non-`NaN` `efficient_frontier/xs`. Readers must clip it at the max Sharpe ratio volatility: keep the points with `xs` ≤ `max_sharpe_ratio/xs[0]` and end the line at the max Sharpe ratio point. Clipping a v31 line changes nothing, so readers may clip every date. Before v31 the efficient frontier ended at the highest simulated return (it could stop short of the highest feasible return, or end in a run of `NaN` points), and the random portfolios ignored the per-ticker weight cap, so some could lie outside the region the optimizer searched.
- **The random portfolios are illustrative only.** They are drawn uniformly from the constraint set. Uniform draws rarely land near the corners of a capped region, where the most extreme portfolios are, so the cloud does not fill the feasible region: it typically stays well inside the efficient frontier and short of its top end. It is not evidence that the frontier is correct. Since v32, `random_portfolios` in `optimization_metadata.yml` quantifies this (see the notes there).
- **Array lengths** (100 and 10,000 for the frontier and the random portfolios) are backend settings, and the capital allocation line has 100 points. Read the shapes from the file and do not hard-code them.
- The backend does no plotting. All visualization of this data is the frontend's responsibility.

### Suggested user-facing wording

The frontend may adapt this wording, but it must keep its meaning. Do not hard-code the constraint values (such as the weight cap) in it. The investment-advice disclaimer is shown once at the bottom of the UI (see "Where the data lives"), so do not repeat it here.

- **Plot title:** "Mean–Variance Portfolio Optimization"
- **Subtitle:** "Expected returns and volatility estimated from daily simple returns; annualized using 252 trading days."
- **Axis labels:** x "Annualized Volatility (σ, %)", y "Annualized Arithmetic Expected Return (%)"
- **Legend labels:**
  - `efficient_frontier`: "Efficient frontier"
  - `tangency_line`: "Capital allocation line (risk-free rate X.XX%)", filling in `risk_free_rate.risk_free_rate` from the same date's `optimization_metadata.yml`, to two decimals
  - `simulated_portfolios`: "Random portfolios"
  - `max_sharpe_ratio`: "Max Sharpe ratio portfolio"
  - `min_var_portfolio`: "Minimum variance portfolio"
- **Caption about the random portfolios:** "Random portfolios are sampled from the same constraints as the optimizer to illustrate the feasible region; they are not evidence that the frontier is correct." Show it with the plot: either near the plot, or inside a collapsible section attached to the plot (such as an accordion), which may be collapsed by default. It must not be left out.

---

## `portfolio_backtest_daily.csv`

**Folder:** `S3_PORTFOLIO_OPTIMIZATION_FOLDER`. **Key:** `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_backtest_daily.csv`. **Format:** CSV with a header row, UTF-8, comma-separated. Read it with the first column as dates, e.g. `pandas.read_csv(..., index_col="date", parse_dates=True)`. *Added in v36; older date folders do not have it, so treat it as optional.*

The realized out-of-sample path of two portfolios over the most recent scored trading days: the maximum Sharpe ratio portfolio, re-optimized on a fixed schedule using only past data, and an equally weighted baseline rebalanced on the same schedule. See `portfolio_backtest_metadata.yml` for the method and the results.

| Column | Type | Meaning |
| --- | --- | --- |
| `date` (first column, the index) | string `YYYY-MM-DD` | NYSE trading day. The first row is the close at which the backtest starts, before any return. |
| `max_sharpe_daily_return`, `equal_weight_daily_return` | float, empty on the first row | The portfolio's simple return on that day, **in percent, not annualized**. |
| `max_sharpe_wealth`, `equal_weight_wealth` | float | Value of the portfolio in **US dollars**, starting from exactly `1.0` on the first row and compounding the daily returns. |

- **Rows:** `evaluation_window.trading_days` + 1. The count is a backend setting, so read it from the metadata or count the rows; do not hard-code it. The dates are consecutive NYSE sessions with no gaps, and every row after the first has values in every column.
- Do not rely on column order. Ignore columns you do not recognize.

---

## `portfolio_backtest_metadata.yml`

**Folder:** `S3_PORTFOLIO_OPTIMIZATION_FOLDER`. **Key:** `<S3_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_backtest_metadata.yml`, in the same date folder as `portfolio_backtest_daily.csv`. **Format:** YAML (block style, keys sorted alphabetically), UTF-8; parse with a safe YAML loader. Dates are quoted strings. *Added in v36; older date folders do not have it, so treat it as optional.*

### How the backtest works

With `t` the date folder's date, `E` scored days (`evaluation_window.trading_days`), `T` training returns (`schedule.training_window_returns`), and a rebalance every `R` sessions (`schedule.rebalance_interval_sessions`). All three are backend settings recorded in this file. Since v46, `E` is the backtest's own setting, independent of the number of scored forecast days (`evaluation_window.target_trading_days` in `all_forecasts_metadata.yml`). Both evaluations end on the date folder's date, but they can start on different dates, so do not assume they cover the same period. Before v46 they were one setting and covered the same dates.

1. There are `E / R` rebalances, at the closes of `t-E`, `t-E+R`, ..., `t-R`. There is no rebalance at `t`.
2. **Max Sharpe ratio portfolio:** at each rebalance, the portfolio is re-optimized for maximum Sharpe ratio on the `T` daily simple returns ending on that date (`T` + 1 closes), under the same constraints as the live optimization. No later price is used. The risk-free rate is the last published DTB3 rate dated **before** the rebalance date, because the same day's rate may not have been available at that close.
3. **Equal-weight baseline:** at each rebalance, the portfolio moves back to `1/N` in each ticker.
4. **Holding:** each portfolio is then held for `R` sessions with no trading, so its weights drift with the tickers' returns and can exceed the weight cap until the next rebalance.
5. **Wealth:** starts at $1.00 at the close of `t-E` and compounds the `E` daily returns.

If an optimization does not converge, or its weights violate the constraints, the portfolio rebalances to the previous target weights instead (equal weights if it is the first rebalance), and that rebalance's `status` says so. If any close or session is missing from the data the backtest needs, the run fails and nothing is uploaded for that date.

The backend checks before uploading that every optimization received exactly `T` returns, that there are exactly `E / R` rebalances and `E` consecutive non-overlapping daily returns, that no training window reaches into its own holding period, that every target weight satisfies the constraints, and that the CSV and YAML files agree.

**The last rebalance is not the live portfolio; `final_optimization` is.** The last rebalance is `R` sessions before the date folder, so `rebalances[-1].max_sharpe.target_weights` come from older data than `optimum_portfolio.weights`, and differ from them. Since v38, the live optimization uses the same training window length and risk-free rule as every rebalance, and it is recorded here as `final_optimization`: one more optimization, at `t`, which holds nothing and adds no return to the backtest. It is copied from the live result, not solved again. The backend checks before uploading that its date, window, risk-free rate, return, volatility, Sharpe ratio, and weights equal those in `optimization_metadata.yml`, and that its return and volatility equal the `max_sharpe_ratio` point in the `.h5` file; if any differ, nothing is uploaded for that date.

### Example

Illustrative values; the structure matches the backend's output. One key per ticker in each weight map, only two shown. Values that depend on the settings are shown as `<placeholders>`: read them from the real file.

```yaml
annualization:
  method: compounded
  periods_per_year: 252
assumptions:
  drift: weights drift with returns between rebalances
  execution: idealized close-to-close rebalancing with no transaction costs or slippage; not an execution-realistic trading simulation
  fallback: if an optimization fails, rebalance to the previous target weights, or to equal weights at the first rebalance
  prices: adjusted closes as returned by Massive.com; dividends are not assumed to be included
  returns: simple daily returns
  risk_free_rate: the last DTB3 rate dated before each rebalance
  universe: a fixed set of tickers, chosen with hindsight
constraints:
  fully_invested: true
  long_only: true
  weight_cap: 40.00000000000001
  weight_cap_multiple: 4
evaluation_window:
  end_date: '2026-10-09'
  initial_wealth_date: '<close before the first scored day, YYYY-MM-DD>'
  start_date: '<first scored day, YYYY-MM-DD>'
  trading_days: <number of scored days>
final_optimization:
  annualized_return: 32.4
  date: '2026-10-09'
  risk: 11.0
  risk_free_rate:
    risk_free_rate: 4.0
    risk_free_rate_date: '<day before, or earlier, YYYY-MM-DD>'
  sharpe_ratio: 2.6
  training_window:
    first_close_date: '<YYYY-MM-DD>'
    first_return_date: '<YYYY-MM-DD>'
    last_close_date: '2026-10-09'
    returns: <training returns>
  weights:
    GLD: 40.0
    NVDA: 3.5e-15
metrics:
  equal_weight:
    annualized_volatility: 11.5
    cagr: 9.1
    cumulative_return: 2.2
    max_drawdown: -4.8
  max_sharpe:
    annualized_volatility: 12.7
    cagr: 14.6
    cumulative_return: 3.3
    max_drawdown: -6.1
rebalances:
- date: '<rebalance date, YYYY-MM-DD>'
  equal_weight:
    in_sample:
      annualized_return: 7.2
      risk: 11.8
      sharpe_ratio: 0.3
    target_weights:
      GLD: 10.0
      NVDA: 10.0
    weights_before_rebalance: null
  holding_period:
    end_date: '<last session held, YYYY-MM-DD>'
    start_date: '<first session held, YYYY-MM-DD>'
  max_sharpe:
    in_sample:
      annualized_return: 33.1
      risk: 11.4
      sharpe_ratio: 2.5
    target_weights:
      GLD: 40.0
      NVDA: 3.2e-14
    weights_before_rebalance: null
  risk_free_rate:
    risk_free_rate: 4.05
    risk_free_rate_date: '<day before, or earlier, YYYY-MM-DD>'
  status: optimized
  training_window:
    first_close_date: '<YYYY-MM-DD>'
    first_return_date: '<YYYY-MM-DD>'
    last_close_date: '<same as date>'
    returns: <training returns>
schedule:
  fallback_count: 0
  rebalance_count: <number of rebalances>
  rebalance_interval_sessions: <sessions between rebalances>
  training_window_returns: <training returns>
tickers:
- XBI
- QQQ
units:
  annualized_volatility: percent_per_year
  cagr: percent
  cumulative_return: percent
  daily_return: percent
  max_drawdown: percent
  risk_free_rate: percent_per_year
  wealth: usd
  weights: percent
```

### Fields

| Path | Type | Meaning |
| --- | --- | --- |
| `tickers` | list of string | Tickers in both portfolios. Order is not significant. |
| `evaluation_window.initial_wealth_date` | string `YYYY-MM-DD` | The close at which wealth is $1.00: the first row of the CSV, and the first rebalance date. |
| `evaluation_window.start_date`, `.end_date` | string `YYYY-MM-DD` | First and last scored day. `end_date` equals the date folder. |
| `evaluation_window.trading_days` | int | Number of scored days. A backend setting, so **read this value wherever user-facing text states it**. |
| `schedule.training_window_returns` | int | Daily simple returns in each training window; the window holds one more close than this. A backend setting. |
| `schedule.rebalance_interval_sessions` | int | Trading sessions between rebalances, and in each holding period. A backend setting. |
| `schedule.rebalance_count` | int | Number of rebalances: `trading_days` / `rebalance_interval_sessions`. |
| `schedule.fallback_count` | int | Rebalances that used the fallback. Normally 0. If it is not, that is an assumption worth including in the section's collapsible accordion (see "Notes"). |
| `constraints.*` | mixed | The same constraints as in `optimization_metadata.yml`, applied to the target weights only. |
| `annualization.method`, `.periods_per_year` | string, int | How `cagr` is annualized: `compounded`, with `252` trading days per year. `metrics.<portfolio>.annualized_volatility` uses the same 252 days but is scaled by √252, as in the optimization files. |
| `units.*` | string | Units of each quantity in the two backtest files. |
| `assumptions.*` | string | The limits of the comparison, in words. |
| `metrics.<portfolio>.cumulative_return` | float | Compounded total return over the scored days, **in percent**: (last wealth − 1) × 100. `<portfolio>` is `max_sharpe` or `equal_weight`. |
| `metrics.<portfolio>.cagr` | float | Compound annual growth rate, **in percent**: (last wealth ^ (252 / `trading_days`) − 1) × 100. |
| `metrics.<portfolio>.max_drawdown` | float | Largest peak-to-trough decline of wealth, **in percent**, zero or negative (−6.1 means a 6.1 % decline). The initial $1.00 counts toward the running peak. |
| `metrics.<portfolio>.annualized_volatility` | float | Realized volatility over the scored days, **annualized, in percent**: the sample standard deviation (n − 1 in the denominator) of the portfolio's daily simple returns in `portfolio_backtest_daily.csv` × √252. The returns' own mean is removed, not the risk-free rate. *Added in v42; absent in older files, so treat it as optional.* |
| `rebalances[].date` | string `YYYY-MM-DD` | The rebalance date, which is the optimization's cutoff. In date order. |
| `rebalances[].status` | string | `optimized`, `fallback_previous_weights`, or `fallback_equal_weights`. Applies to `max_sharpe` only: the baseline needs no optimization. |
| `rebalances[].training_window.first_close_date`, `.first_return_date`, `.last_close_date` | string `YYYY-MM-DD` | The first close, the first daily return (one session later), and the last close of the training window. `last_close_date` equals `date`. |
| `rebalances[].training_window.returns` | int | Daily returns in the window. Always equals `schedule.training_window_returns`. |
| `rebalances[].holding_period.start_date`, `.end_date` | string `YYYY-MM-DD` | First and last session of the holding period after this rebalance. The periods are consecutive and do not overlap. |
| `rebalances[].risk_free_rate.risk_free_rate` | float | The DTB3 rate this optimization used, **annualized, in percent**. |
| `rebalances[].risk_free_rate.risk_free_rate_date` | string `YYYY-MM-DD` | Observation date of that rate, always before `date`. |
| `rebalances[].<portfolio>.target_weights` | map ticker → float | Weights set at this rebalance, **in percent**, one key per ticker. The weight-noise rules of `optimization_metadata.yml` apply. |
| `rebalances[].<portfolio>.weights_before_rebalance` | map ticker → float, or null | The drifted weights held just before this rebalance, in percent. `null` on the first rebalance, when nothing was held. |
| `rebalances[].<portfolio>.in_sample.annualized_return`, `.risk`, `.sharpe_ratio` | float | The target weights evaluated **on this rebalance's training window**, in the conventions of `optimization_metadata.yml`: annualized arithmetically, in percent, and a dimensionless annualized Sharpe ratio. These are in-sample estimates, not realized results. *Added in v38; absent in v36 and v37 files, so treat them as optional.* |
| `final_optimization.date` | string `YYYY-MM-DD` | The live optimization's date: the date folder. *Added in v38; absent in older files, so treat the whole section as optional.* |
| `final_optimization.training_window.*` | mixed | Its training window, in the form of `rebalances[].training_window`. `first_close_date` equals `date_updated.date_from`. |
| `final_optimization.risk_free_rate.*` | mixed | Its risk-free rate, in the form of `rebalances[].risk_free_rate`. |
| `final_optimization.annualized_return`, `.risk`, `.sharpe_ratio` | float | Equal to the same keys of `optimum_portfolio` in `optimization_metadata.yml`. |
| `final_optimization.weights` | map ticker → float | Equal to `optimum_portfolio.weights`, in percent. |
| `computed_at` | string | When the backend computed this date folder, in UTC, as `YYYY-MM-DDTHH:MM:SSZ` (for example `2026-11-14T12:00:00Z`): the start of the run that wrote it. All metadata files from one run have the same value. It can be long after the folder's date (see "Re-runs and past dates"). *Added in v48; absent in older files, so treat it as optional.* |
| `contract_version` | int | The version of this contract the date folder was written under (`48` or later). Files without it were written under an earlier version. *Added in v48; absent in older files, so treat it as optional.* |

### Notes

**Where this text goes.** The wording in these notes is suggested, and nothing here requires text at a specific place on the page. Put assumptions and limitations in the collapsible accordion that explains the assumptions of the section showing the backtest. For disclaimers, rely on the single "Not investment advice" text at the bottom of the UI, which already covers every file in this contract (see "Where the data lives"), and do not add another.

- **Different annualization from the optimization files.** `cagr` is **compounded** and realized, while `annualized_return` in `optimization_metadata.yml`, `final_optimization`, and `in_sample` is arithmetic and is an in-sample estimate, not a realized return. Do not put them side by side as the same quantity, and label `cagr` "CAGR".
- **`cagr` extrapolates.** It scales the window to a full year through the exponent 252 / `trading_days`. For a window much shorter than a year, a few good days give a large number. For a window close to a year, `cagr` is close to `cumulative_return`, and the two differ only by the remaining scaling. Prefer `cumulative_return` for the headline comparison.
- **Realized versus estimated volatility** (since v42). `metrics.<portfolio>.annualized_volatility` is realized over the scored days, out of sample. `risk` in `optimization_metadata.yml`, `final_optimization`, and `in_sample` is an in-sample estimate from the training window. Do not present them as the same quantity. Scaling by √252 assumes daily returns are serially uncorrelated, and with only `trading_days` returns the value is itself noisy. Label it "Annualized volatility (realized)" or similar.
- **Small sample.** One window of `trading_days` days cannot show that either portfolio is better, even when it is about a year long: the difference is one noisy outcome of one market period, and the backtest has only `rebalance_count` rebalancing decisions.
- **Fixed universe.** The tickers are the ones in the portfolio today, chosen with hindsight, for both portfolios.
- **Idealized execution.** Each rebalance happens exactly at the close whose price the optimization just used, with no transaction costs or slippage. The max Sharpe ratio portfolio generally trades more than the baseline, so costs would hurt it more. This is not an execution-realistic trading simulation.
- **Prices.** The returns come from Massive.com's adjusted closes, and dividends are not assumed to be included (see "Shared conventions"). If they are not, these are price returns, which understate the tickers that pay dividends.
- **Comparing dates.** Consecutive date folders overlap almost entirely, and the settings can differ between folders. Compare `trading_days` and `schedule` before comparing metrics across dates. In particular, date folders written before v46 have backtests of the same length as their forecast evaluation, which is much shorter than in later folders.
- **Not the forecasts' period** (since v46). The backtest and the forecast metrics end on the same date but generally start on different dates. Do not label them as covering the same period, and do not assume they share an x-axis range.
- **Suggested wording for the accordion** (fill in `{n}` from `evaluation_window.trading_days`, `{r}` from `schedule.rebalance_interval_sessions`, and `{t}` from `schedule.training_window_returns`): "Growth of $1 over the last {n} trading days. Every {r} trading days, the max Sharpe ratio portfolio is re-optimized on the previous {t} daily returns, using only data available at that date, and the baseline is reset to equal weights; between those dates no trades are made and weights drift. Execution is idealized, with no transaction costs. This is one period of {n} trading days for a fixed set of tickers chosen with hindsight, so it does not show that either portfolio is better."

---

## `all_forecasts.csv`

**Folder:** `S3_TIME_SERIES_FOLDER`. **Key:** `<S3_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv`. **Format:** CSV with a header row, UTF-8, comma-separated, written by `pandas.DataFrame.to_csv(index=True)` (the index is the first column). Read it with the first column parsed as dates, e.g. `pandas.read_csv(..., index_col="pred_date", parse_dates=True)`.

### Shape

A wide table with one row per forecast date and, for every ticker `T`, six columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `pred_date` (first column, the index) | timestamp string, e.g. `2026-10-09 17:00:00` | The NYSE trading day being forecast. The time of day (`17:00:00`) is a fixed label with no meaning; use only the date. |
| `T_arima` | float, may be empty | The ARIMA model's one-step-ahead forecast of `T`'s adjusted close (US dollars) for `pred_date`. Empty when `T`'s close on the previous session is missing from the data (since v20; see "Missing data"). |
| `T_arima_p` | integer, may be empty | The autoregressive order `p` of the model that made this row's `T_arima` forecast. Present on every row where `T_arima` is, including the last; empty where `T_arima` is. *Added in v16.* |
| `T_arima_d` | integer, may be empty | The differencing order `d` of that model, stated on **log prices**. Always `1` where `T_arima` has a value (including the last row); empty where `T_arima` is. *Added in v18.* |
| `T_arima_q` | integer, may be empty | The moving-average order `q` of that model. Present on every row where `T_arima` is, including the last; empty where `T_arima` is. *Added in v16.* |
| `T_naive` | float | The **naive (random walk) baseline** forecast of `T`'s adjusted close (US dollars) for `pred_date`: the most recent observed close before `pred_date`, which is the previous trading day's close unless that day is missing from the data. It uses exactly the data the row's ARIMA model saw (never `pred_date`'s own close), so the two forecasts are directly comparable. `T_arima` / `T_naive` = exp(ARIMA's forecast log return). Present on every row, including the last. *Added in v19.* |
| `T` | float, may be empty | The actual adjusted close (US dollars) of `T` on `pred_date`. **Empty on rows that have no actual value yet** (see below). |

Each model is fit as ARIMA(p, 0, q) on daily **log returns**, which is the same model as ARIMA(p, 1, q) on log prices. The three order columns describe it on log prices: `(T_arima_p, T_arima_d, T_arima_q)` is the `(p, d, q)` of ARIMA(p, d, q) on log prices, so `T_arima_d` is always `1` where present. Orders `p` and `q` are re-selected for every row and ticker (see "How the forecasts are produced"), so they can change from row to row. With the current settings each is 0 or 1, and `(p, q) = (0, 0)` never occurs. Do not hard-code that range, and read `d` from the column rather than assuming 1. Date folders written before v16 have no order columns, and those written before v18 have no `T_arima_d` column, so treat all three as optional.

Example header for three tickers (illustrative):

```
pred_date,GLD_arima,GLD_arima_p,GLD_arima_d,GLD_arima_q,GLD_naive,GLD,NVDA_arima,NVDA_arima_p,NVDA_arima_d,NVDA_arima_q,NVDA_naive,NVDA,QQQ_arima,QQQ_arima_p,QQQ_arima_d,QQQ_arima_q,QQQ_naive,QQQ
```

Do not rely on column order. Identify columns by **exact suffix match**:

- Forecast columns end with `_arima`.
- Order columns end with `_arima_p`, `_arima_d`, or `_arima_q`.
- Naive-baseline columns end with `_naive` (since v19; absent in older date folders, so treat them as optional).
- Actual-price columns are the remaining names that match a ticker exactly.

Do **not** treat every column *containing* `_arima` as a forecast column, because the order columns also contain it.

### Suggested user-facing wording

The frontend may adapt this wording, but it must keep its meaning: the model is fit on log returns, the orders are stated on log prices, and the orders can change daily. It assumes readers know what ARIMA(p, d, q) means.

**Tooltip on a forecast point** (fill in the row's `T_arima_p`, `T_arima_d`, `T_arima_q`):

> ARIMA({p}, {d}, {q}) on log prices
> (fit as ARIMA({p}, 0, {q}) on daily log returns)

**Text near the forecast plot** (fill in `{w}`, the number of closes in each training window, as `forecast_window.training_window_returns` + 1 from the same date's `all_forecasts_metadata.yml`; if that file is absent, as for dates before v21, leave out the parenthesis):

> **How these forecasts are made.** Each point is a one-trading-day-ahead forecast of the closing price, made only with data available before that day. For every day and ticker, an ARIMA model is refit on a window of the most recent closes before that day ({w} trading days), and its orders p and q are chosen again, so they can change from one day to the next. Hover over a forecast to see the model behind it.
>
> **Why d = 1.** The model is fit to daily log returns as ARIMA(p, 0, q). Turning prices into returns is one round of differencing, so the same model on log prices is ARIMA(p, 1, q). The tooltip shows the log-price version.
>
> **Naive forecast:** today's close = yesterday's close. A model adds value only if its errors are smaller than the naive baseline.

### How the forecasts are produced

- **Horizon:** one business day ahead. Each row's forecast was made using only data **before** `pred_date`.
- **Walk-forward:** for each forecast date, an ARIMA model is refit on a sliding window of the closes ending on the last session before `pred_date`. The window length is a backend setting that can change between runs, so **do not assume it**: read `forecast_window.training_window_returns` (daily log returns per window; closes = that value + 1) and `forecast_window.training_window_start_date` from the same date's `all_forecasts_metadata.yml`. Orders `p` and `q` are chosen from 0 to 1 and 0 to 1 (excluding 0,0): since v24, each candidate is fit once on the whole window and the one with the **lowest BIC** wins, and its model makes the forecast. Exactly equal BICs (in practice only when fits fail) go to fewer parameters (`p + q`), then to the lower `p`, so the choice is reproducible. A candidate whose fit fails ranks last; if every candidate's fit fails, the run fails instead of uploading. The window slides one business day per row. Each window contains only NYSE trading sessions (since v15), so one model time step is one session; NYSE holidays are skipped, not treated as missing observations. Each model includes a constant, so its forecast includes the window's average daily log return (a drift term).
- **Before v24** the training window was much shorter, and orders were scored by one-step-ahead RMSE and BIC rank on rolling sub-windows of it (with the v17 and v23 rules for failed fits and ties). For dates from v21 on, the metadata records the window (`training_window_returns`) and sub-window (`models.arima.order_selection_subwindow_returns`) used; the changelog gives the earlier values.
- **Back to dollars:** the model forecasts a log return. The backend converts it to a price with `last_price × exp(forecasted log return)`, so `T_arima` is in the same units as `T`.
- **Last row:** the final row is the first NYSE trading day after the latest available data. Its `T` values are **empty** because that day has not happened yet. Treat that row as the live forecast and the rows above it as the backtest.
- **Rows** are unique by `pred_date` and sorted ascending. There is normally one backtest row with an actual close per scored day, plus the live row, so normally `evaluation_window.target_trading_days` + 1 rows. The count is a backend setting and can be lower when data is missing, so **do not hard-code it**: count the rows of the CSV (the backtest is the rows with a non-empty actual close), and read `target_trading_days` from the metadata for the number that was scored.
- **Missing data** (since v20). Forecasts are made only from closes that are present. If a whole session is absent from the data, there is no row for the session after it, because a one-step-ahead forecast for that day would need the missing close; the row for the missing session itself has an empty `T`. If only one ticker's close is missing on a session, the next row keeps its `pred_date`, but that ticker's `T_arima` and order columns are empty, while its `T_naive` still uses the most recent observed close. Rows are always unique by `pred_date`. Because the order columns can be empty, a CSV reader may load them as floats; treat their non-empty values as whole numbers.

### Not included

The file does **not** contain error metrics or confidence intervals. The naive baseline is included since v19 (`T_naive`). The selected model orders are included: `T_arima_p` and `T_arima_q` since v16, `T_arima_d` since v18. Since v21, error metrics for both forecasts are in `all_forecasts_metadata.yml` (next section). Use those rather than computing metrics in the frontend, so every surface reports the same numbers.

---

## `all_forecasts_metadata.yml`

**Folder:** `S3_TIME_SERIES_FOLDER`. **Key:** `<S3_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts_metadata.yml`, in the same date folder as `all_forecasts.csv`. **Format:** YAML (block style, keys sorted alphabetically), UTF-8; parse with a safe YAML loader. *Added in v21; date folders written before v21 do not have it, so treat it as optional.*

It describes the most recent forecasts in that date's `all_forecasts.csv`: the training window of the live forecast, and how accurate the ARIMA and naive forecasts were over the last `evaluation_window.target_trading_days` trading days with an actual close. Both window lengths are backend settings that can change between runs; this file records the values each run used, so read them here rather than assuming them. The backend computes it from the CSV exactly as uploaded.

**All keys are lowercase snake_case, including the ticker keys under `ticker_metrics`.** Ticker symbols are lowercased here (`xbi`, `qqq`, ...), unlike every other file, where they are uppercase. Match them with a case-insensitive comparison, e.g. `ticker.lower()`.

### Example

Illustrative values; the structure matches the backend's output. Values that depend on the window lengths are shown as `<placeholders>`, because those lengths are backend settings: read them from the real file.

```yaml
evaluation_window:
  end_date: '2026-10-09'
  start_date: '<first scored session, YYYY-MM-DD>'
  target_trading_days: <number of scored days>
forecast_window:
  forecast_date: '2026-10-12'
  horizon_trading_days: 1
  training_window_end_date: '2026-10-09'
  training_window_returns: <daily log returns per training window>
  training_window_start_date: '<first session of the live training window, YYYY-MM-DD>'
models:
  arima:
    d: 1
    description: ARIMA(p, 0, q) on daily log returns, which is ARIMA(p, 1, q) on log prices
    excluded_orders:
    - - 0
      - 0
    order_selection: lowest BIC, each candidate order fit once on the whole training window
    order_selection_subwindow_returns: <same as training_window_returns>
    order_selection_tie_break: fewer parameters (p + q), then lower p
    p_values:
    - 0
    - 1
    q_values:
    - 0
    - 1
  naive:
    description: the most recent observed close (random walk)
ticker_metrics:
  xbi:
    arima:
      directional_accuracy: 0.55
      mae: 1.21
      mape: 1.32
      n_directional: <scored days with a nonzero move>
      rmse: 1.52
    evaluation_end_date: '2026-10-09'
    evaluation_start_date: '<first scored session, YYYY-MM-DD>'
    n_evaluated: <normally target_trading_days>
    naive:
      mae: 1.25
      mape: 1.37
      rmse: 1.58
    relative_mae: 0.968
    skill: 0.032
units:
  directional_accuracy: fraction
  mae: usd_per_share
  mape: percent
  relative_mae: ratio
  rmse: usd_per_share
  skill: ratio
```

### Fields

| Path | Type | Meaning |
| --- | --- | --- |
| `forecast_window.training_window_start_date` | string `YYYY-MM-DD` | First trading session of the live forecast's training window. |
| `forecast_window.training_window_end_date` | string `YYYY-MM-DD` | Last session of that window, the latest close used. Equals the date folder. |
| `forecast_window.forecast_date` | string `YYYY-MM-DD` | The date of the live forecast: the CSV's last `pred_date`, one NYSE trading day after `training_window_end_date`. |
| `forecast_window.training_window_returns` | int | Daily log returns in each training window; the window holds one more close than this. A backend setting that can change between runs, so read it from each file and **use it wherever user-facing text states the training window length** (see the suggested wording for the forecast plot). Also use it to tell methods apart when comparing dates. |
| `forecast_window.horizon_trading_days` | int | Forecast horizon (`1`: one trading day ahead). |
| `evaluation_window.target_trading_days` | int | Number of most recent days with an actual close that are scored. A backend setting that can change between runs, so **read this value wherever user-facing text states the number of scored days** (see the suggested wording below) instead of writing a fixed number. |
| `evaluation_window.start_date`, `.end_date` | string `YYYY-MM-DD` or null | Earliest start and latest end of the per-ticker evaluation windows. Normally the same for every ticker. |
| `units.*` | string | Units of each metric: `usd_per_share` (RMSE, MAE), `percent` (MAPE), `ratio` (relative MAE, skill), `fraction` (directional accuracy, from 0 to 1). |
| `models.arima.*`, `models.naive.description` | mixed | How each forecast is made: the ARIMA order search (`p_values`, `q_values`, `excluded_orders`), `d`, the order-selection rule and its sub-window length, the tie-break rule (`order_selection_tie_break`, since v23; absent in older files), and the naive rule. These are taken from the backend's settings. Since v24, candidates are scored on one window, the whole training window, so `order_selection_subwindow_returns` equals `forecast_window.training_window_returns` (it was the rolling sub-window length before). |
| `ticker_metrics.<ticker>.n_evaluated` | int | Days actually scored. Normally equal to `evaluation_window.target_trading_days`; fewer if a ticker's forecast or close was missing on some of those days. Both models are always scored on the same days. |
| `ticker_metrics.<ticker>.evaluation_start_date`, `.evaluation_end_date` | string `YYYY-MM-DD` or null | First and last of this ticker's `target_trading_days` most recent days with an actual close. |
| `ticker_metrics.<ticker>.arima.rmse`, `.naive.rmse` | float or null | Root mean squared error of the forecast against the actual close, **US dollars per share**. |
| `ticker_metrics.<ticker>.arima.mae`, `.naive.mae` | float or null | Mean absolute error, **US dollars per share**. |
| `ticker_metrics.<ticker>.arima.mape`, `.naive.mape` | float or null | Mean absolute percentage error, **in percent** (1.3 means 1.3 %). Scale-free, so it can be compared across tickers. |
| `ticker_metrics.<ticker>.relative_mae` | float or null | ARIMA MAE ÷ naive MAE over the same days. **Below 1 means ARIMA was more accurate than the naive forecast.** `null` if naive MAE is 0 or nothing was scored. |
| `ticker_metrics.<ticker>.skill` | float or null | `1 − relative_mae`. **Above 0 means ARIMA beat the naive forecast**, 0 means no better, below 0 means worse. `null` when `relative_mae` is. |
| `ticker_metrics.<ticker>.arima.directional_accuracy` | float or null | Fraction of scored days on which ARIMA predicted the direction of the move from the previous close correctly (0 to 1). Days with no move are skipped; `null` if no day moved. The naive forecast predicts no move, so it has no directional accuracy. |
| `ticker_metrics.<ticker>.arima.n_directional` | int | Days scored for directional accuracy. |
| `computed_at` | string | When the backend computed this date folder, in UTC, as `YYYY-MM-DDTHH:MM:SSZ` (for example `2026-11-14T12:00:00Z`): the start of the run that wrote it. All metadata files from one run have the same value. It can be long after the folder's date (see "Re-runs and past dates"). *Added in v48; absent in older files, so treat it as optional.* |
| `contract_version` | int | The version of this contract the date folder was written under (`48` or later). Files without it were written under an earlier version. *Added in v48; absent in older files, so treat it as optional.* |

### Notes

- **Comparing tickers.** RMSE and MAE are in each ticker's own price units, so a $1 error on a $500 stock and on a $30 stock are not alike. Compare tickers with `mape`, `relative_mae`, or `skill`, which are scale-free.
- **What "relative MAE" is.** It is the ratio of out-of-sample MAEs over the same scored days. It is related to, but not the same as, Hyndman's MASE, which scales by the naive forecast's in-sample error. Label it "relative MAE" (or "skill" for `1 − relative_mae`), not "MASE".
- **Small samples.** The scored days (`target_trading_days`) are a modest sample, so the metrics are noisy and can change from day to day. Consecutive forecasts are not independent, which makes the effective sample smaller still. Say so in user-facing text.
- **Comparing dates.** The number of scored days and the training window can differ between date folders, so their metrics can come from different sample sizes and methods. Compare each file's `target_trading_days` and `training_window_returns` before comparing metrics across dates.
- **Suggested wording near the metrics** (fill in `{n}` from `evaluation_window.target_trading_days`): "Accuracy over the last {n} trading days, comparing each forecast with the actual close. Relative MAE below 1 (skill above 0) means ARIMA's errors were smaller than the naive forecast's. RMSE and MAE are in dollars per share; MAPE is in percent and can be compared across tickers. {n} trading days is a modest sample, so treat these as indicative."

---

## Reader requirements (for the frontend)

1. **Ignore unknown things.** Unknown YAML keys, HDF5 groups or datasets, and CSV columns must be ignored without error. This is what makes additive changes safe.
2. **Do not hardcode** the ticker list, array lengths, row counts, or the training, scoring, and rebalancing window lengths. Read the window lengths from `all_forecasts_metadata.yml` (`forecast_window.training_window_returns`, `evaluation_window.target_trading_days`) and `portfolio_backtest_metadata.yml` (`schedule`, `evaluation_window.trading_days`), and the rows from `all_forecasts.csv` and `portfolio_backtest_daily.csv`.
3. **Handle `NaN` and empty cells** as described above.
4. **Show data dates.** Display the selected date (equal to `date_updated.date_to`) so users can tell how current the data is.
5. **Fail soft.** If a file is missing or does not parse, show an explanatory message instead of crashing. Missing or partial data is expected while a run is mid-upload.
6. **Label units.** Daily versus annualized, and percent versus dollars, must be stated wherever a value is shown.
7. **Date selection.** Offer only *available* dates (see "Which dates are available"), newest first, with the newest selected by default. Load all files from the **same** date in both folders; never mix dates across files. Ignore keys whose segment after the folder is not a valid date, including stale objects directly under a folder from contract v8 and earlier.
8. **Do not assume consecutive dates.** Runs are manual, so dates have gaps (weekends, holidays, days not run).

## Verification status

Checked against real backend output (2026-10-11), unless dated otherwise:

- Storage layout (2026-10-12):
  - **Console (the human):** the account has exactly one Space (bucket), whose name is the value of `S3_BUCKET_NAME`, in region `nyc3`.
  - **Programmatic listing** with the recommended configuration (endpoint `https://nyc3.digitaloceanspaces.com`, `ListObjectsV2(Bucket=<space>, Prefix="<folder>/")`) returned exactly these keys:
    - `time-series/2026-10-09/all_forecasts.csv`
    - `portfolio-optimization/2026-10-09/optimization_metadata.yml`
    - `portfolio-optimization/2026-10-09/portfolio_optimization_plot_data.h5`
    - `portfolio-optimization/optimization_metadata.yml`, a stale pre-v9 object directly under the folder, which readers ignore. The human has since deleted it.

    A later listing after the backend switched to `Bucket=<space>` with the folder in the key showed the same three dated keys and no stale objects.

    So the folder values were `time-series` and `portfolio-optimization` at the time.
  - **Folder names as buckets fail:** `ListObjectsV2` with a folder name as `Bucket` fails, with `NoSuchBucket` from the backend's configuration and `NoSuchKey` from the frontend's.

- `optimization_metadata.yml` (v5 output): every key in the example above, including the v4 additions, quoted date strings, and floating-point noise in weights. The values are internally consistent: `sharpe_ratio` = (`annualized_return` − `risk_free_rate`) / `risk` = (34.343 − 4.05) / 12.440 = 2.435.
- `portfolio_optimization_plot_data.h5` (v5 output, `h5dump -A`): the five root attributes with the documented values and types (variable-length UTF-8 strings; `periods_per_year` is `H5T_STD_I64LE`), plus group names, dataset names, dtype (`H5T_IEEE_F64LE`), and shapes (100, 1, 1, 10000, 100).
- `all_forecasts.csv`: first column `pred_date`, the `T_arima` / `T` naming, and empty actuals in the final row.
- The "How the forecasts are produced" description matched the backend's defaults at the time (the pre-v24 method). The current method (v24+) is checked against the code and unit tests (`tests/test_arima_models.py`, `tests/test_ticker_predict.py`), not yet against real output.
- `.h5` values (v5 output, `h5dump -d`): `max_sharpe_ratio` = (12.4398, 34.3434) and `min_var_portfolio` = (6.89459, 12.8517) equal the YAML `risk` / `annualized_return` of the same run. `tangency_line/ys` starts at 4.05 (= `risk_free_rate`), is evenly spaced (a straight line), and passes through the max Sharpe ratio point with slope `sharpe_ratio`. (That was the pre-v31 line, which continued past the point.)
- The annualization formulas and the straight capital allocation line are covered by the backend's unit tests (`tests/test_annualization.py`).
- `portfolio_backtest_metadata.yml` (v36 output, 2026-10-31, from a real run without upload): the rebalance count, no fallbacks, the training window length and dates of every rebalance, risk-free rates dated before their rebalances, non-overlapping holding periods, an evaluation start date equal to that of `all_forecasts_metadata.yml` (expected before v46 only, when the two lengths were one setting), equal-weight target weights, and maximum drawdowns of zero or below. The backtest's unit tests passed the same day.
- The v38 and v39 backtest and optimization output (2026-11-01, data through 2026-10-30, read from the local files of a run with uploads enabled):
  - `final_optimization` is identical to `optimum_portfolio` in return, volatility, Sharpe ratio, and all weights; its window starts on `date_updated.date_from`, and its risk-free rate and date equal those of `optimization_metadata.yml`.
  - `max_return_portfolio` has the closed-form return and weights.
  - `portfolio_backtest_daily.csv` has `trading_days` + 1 rows, wealth of exactly `1.0` on the first row with empty returns, and dates matching `evaluation_window`.
  - `cumulative_return`, `cagr`, and `max_drawdown` of both portfolios, recomputed by hand from the CSV's wealth, equal the metadata.
  - `in_sample` holds on all rebalances: the max Sharpe ratio portfolio's Sharpe ratio is at least the baseline's, and the Sharpe ratio formula was recomputed by hand for one rebalance.

**v31 plot geometry: covered by unit tests on synthetic prices, and checked by the backend on every run before upload, but not yet verified against real output.** The unit tests are `tests/test_problem.py`, `tests/test_portfolio.py`, `tests/test_workflow.py`, and `tests/test_pipeline.py`:
- Every random portfolio satisfies the shared constraints, including a binding weight cap.
- No random portfolio has a higher Sharpe ratio than the max Sharpe ratio point or lies left of the efficient frontier.
- The minimum variance volatility equals the frontier's smallest volatility.
- The frontier runs from the minimum variance return to the highest feasible return, which the tests check against a closed-form solution.
- The capital allocation line starts at (0, risk-free rate), is straight, and ends exactly at the max Sharpe ratio point.
- The `.h5` max Sharpe ratio and minimum variance points equal the YAML values.
- Every `.h5` volatility and return equals the backend's single statistics function applied to the stored weights.
- (v32) The top of the frontier equals the closed-form maximum feasible return, also checked on every run.
- (v32) The random portfolios' mean return matches the equal-weight return.
- (v32) Their spread matches sqrt(Σ(Rᵢ − R̄)² / (n (n + 1))) when the cap does not bind, and halves under a cap that leaves the middle triangle (3 tickers, 50 %).
- (v32) Draws near the best ticker's corner are 1/32 of all draws with 3 tickers and essentially none with 10.

**Not yet** verified against real data (derived from the code):

- That the date folder equals `date_updated.date_to`, and the upload order (`all_forecasts.csv`, `all_forecasts_metadata.yml`, `.h5`, then `optimization_metadata.yml`). The construction of the date part of the key and of local paths is covered by unit tests (`tests/test_layout.py`).
- The v8 NYSE calendar behavior on real data (covered by unit tests in `tests/test_date_manager.py`).
- `all_forecasts_metadata.yml` (v21) on real output. The metric formulas are covered by unit tests on hand-computed tables (`tests/test_metrics.py`), and the file's windows and upload order by `tests/test_pipeline.py`.
- `NaN` entries in `efficient_frontier/xs` (inferred from the code for infeasible target returns). Readers must handle `NaN` regardless.
- The match between `final_optimization` and the `.h5` file on real output, by reading the file directly. It is covered by unit tests and checked on every run before upload, and a real run passed that check.
- The objects in the bucket for the backtest files. Only the local files were read.
- `metrics.<portfolio>.annualized_volatility` (v42) on real output. It is covered by unit tests (`tests/test_backtest.py`) and recomputed from the wealth on every run before upload.
- The temporal-leakage, missing-data, weight-drift, and fallback behavior of the backtest, which are covered by unit tests on synthetic prices (`tests/test_backtest.py`, `tests/test_pipeline.py`). Each run also checks the structure of the backtest and that its two files agree before uploading.

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| 48 | 2026-11-14 | **Additive.** Each of the three metadata files (`optimization_metadata.yml`, `portfolio_backtest_metadata.yml`, and `all_forecasts_metadata.yml`) gains two top-level keys: `computed_at` (string, UTC, `YYYY-MM-DDTHH:MM:SSZ`), when the run that wrote the date folder started, the same in all three; and `contract_version` (int), the version of this contract the folder was written under. No existing key, column, file, value, unit, or upload order changes. They make a date folder computed after its date (v47) recognizable. The backend can now also backfill many past dates in one command, writing each date as in v47. Frontend: no change required; treat both keys as optional, since older files lack them; optionally show when a date was computed. |
| 47 | 2026-11-14 | Behavior change in which date folders the backend writes; no change to files, keys, columns, types, units, values, or the upload order. The backend can now be run for a past NYSE session (an as-of mode): it uses only prices up to that date, as if the data ended then, and writes (and uploads) that date's folder, replacing any files already in it. A run still writes only its own date folder and never deletes any. "The date prefix" replaces the sentence "Older dates are never modified or deleted by the backend" with "Re-runs and past dates": a date folder can be computed after its date, with the then-current method, settings, and adjusted prices, and can be rewritten. Frontend: no change required; do not assume a date folder was computed on its date or never changes, and keep listing dates on every load, since past dates can be added. |
| 46 | 2026-11-12 | **Configuration change**; no key, column, file name, type, unit, or upload order changes. The portfolio backtest now has its own number of scored days, set independently of the forecasts': the backend's `portfolio_backtest.evaluation_window_length` setting, now 250 trading days (about a year), with the rebalance interval unchanged. Before v46 the backtest scored the same days as the forecasts (60). The forecast files are unchanged: `all_forecasts.csv` and `all_forecasts_metadata.yml` still score the last `target_trading_days` days. In the backtest files, values change but keys do not: `evaluation_window.trading_days` and `start_date` reflect the longer window, `schedule.rebalance_count` grows accordingly, `rebalances` has one entry per rebalance, and `portfolio_backtest_daily.csv` has `trading_days` + 1 rows. The metrics are computed over the longer window, so `cagr` extrapolates much less and is close to `cumulative_return`. Both evaluations still end on the date folder's date. "How the backtest works" no longer says both evaluations cover the same dates; the notes on CAGR, small samples, and comparing dates, and the suggested accordion wording, are updated, and a note says the backtest and forecasts can cover different periods. The backend now downloads about two years of prices for the backtest. Date folders written before v46 keep their shorter backtests. Frontend must: not assume the backtest and the forecast metrics share a start date or x-axis range; not assume a backtest row count or about three months of backtest; keep filling `{n}` in the backtest text from `evaluation_window.trading_days`, and use the updated accordion wording. |
| 45 | 2026-11-07 | **Configuration rename; no objects move and key strings are unchanged.** The environment variables shared by both sides are renamed to drop the old project prefix: `FSF_FRONT_END_BUCKET_NAME` is now `S3_BUCKET_NAME`, `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER` is now `S3_PORTFOLIO_OPTIMIZATION_FOLDER`, and `FSF_FRONT_END_TIME_SERIES_FOLDER` is now `S3_TIME_SERIES_FOLDER`. Values are unchanged. No fallback: the backend reads only the new names. Its own credential, region, and endpoint variables now use boto3's standard names (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL_S3`); they are not part of the contract, and "Connecting" now says so. File names, formats, keys, values, and the upload order are unchanged. Frontend must: read the three new names, rename its Hugging Face secrets and any local `.env` (same values), and drop the old names. Nothing breaks at run time if one side renames first, because each side reads its own environment. Optionally, use the `AWS_*` names for its own credentials. |
| 44 | 2026-11-07 | Documentation only; no change to files, keys, attributes, columns, types, units, values, or the bucket. The frontend repository is renamed from `fuzzy-system-finance` to `aliciadata-finance-frontend`, so the title and "Source of truth" now use the new name. Frontend: no change required; replace its copy with this file. |
| 43 | 2026-11-07 | Documentation only; no change to files, keys, attributes, columns, types, units, values, or the bucket. The backend repository is renamed from `fuzzy-system-research` to `aliciadata-finance-backend` ("AliciaData Finance Backend"), so the title and "Source of truth" now use the new name. Environment variable names, folder names, and object keys are unchanged. Frontend: no change required; replace its copy with this file. |
| 42 | 2026-11-01 | **Additive.** `portfolio_backtest_metadata.yml` gains `metrics.<portfolio>.annualized_volatility` for `max_sharpe` and `equal_weight`: the realized volatility of the portfolio's daily simple returns over the scored days, annualized (sample standard deviation × √252), in percent, and `units.annualized_volatility` (`percent_per_year`). No existing key, column, file, value, unit, or upload order changes. The backend recomputes the value from the wealth series before uploading. A backtest now needs at least 2 scored days, because a standard deviation is undefined for one return; the current setting is well above that. Date folders written before v42 do not have the key unless re-run. Frontend: no change required; optionally show the value; treat it as optional for older dates; do not present it as the same quantity as the in-sample `risk` fields. |
| 41 | 2026-11-01 | Documentation only; no change to files, keys, attributes, columns, types, units, or values. The backtest text added since v36 no longer asks for text at specific places on the page. Its wording is now suggested: assumptions and limitations go in the collapsible accordion that explains the assumptions of the section showing the backtest, and disclaimers rely on the single "Not investment advice" text at the bottom of the UI, with no additional disclaimer. Frontend: no required change; use the suggested wording in the accordion as you see fit. |
| 40 | 2026-11-01 | Documentation only; no change to files, keys, attributes, columns, types, units, or values. "Verification status" now records that `final_optimization`, `in_sample`, `max_return_portfolio`, `portfolio_backtest_daily.csv`, and the backtest metrics were checked against the local files of a real run. Still listed as not verified: reading the `.h5` match directly, and the backtest objects in the bucket. Frontend: no change required; replace its copy with this file. |
| 39 | 2026-11-01 | Bug fix; no change to files, keys, columns, types, or units. `max_return_portfolio` is now the exact closed-form solution instead of a numerical optimization. The numerical optimizer could report success while falling slightly short of the true maximum when tickers had nearly equal mean returns, and the backend's check then failed the run before uploading. In new date folders, `max_return_portfolio` and the top of `efficient_frontier` can differ very slightly from what the optimizer gave, and its weights are exact. A run can no longer fail because the maximum return optimization did not converge, so "Converged optimizations only" now names only the max Sharpe ratio and minimum variance optimizations. Frontend: no change required; replace its copy with this file. |
| 38 | 2026-11-01 | **Configuration and methodology change, plus additive keys, approved by the human.** No key, column, file, type, or unit is renamed or removed, and the upload order is unchanged. Purpose: the live max Sharpe ratio portfolio now appears in the backtest file and is checked against the optimization files. (1) The live optimization now uses the backtest's training window length, one return more than before, so `date_updated.date_from` moves one session earlier and every value in `optimization_metadata.yml` and `portfolio_optimization_plot_data.h5` shifts slightly. (2) The live risk-free rate now follows the backtest's rule, the last DTB3 rate dated before the date folder's date, so it no longer depends on the day the backend ran; a run made after the as-of date's rate was published uses the rate before it. `risk_free_rate_date` keeps its format. (3) `portfolio_backtest_metadata.yml` gains `final_optimization` (the live optimization's date, window, risk-free rate, return, volatility, Sharpe ratio, and weights) and `rebalances[].<portfolio>.in_sample` (each portfolio's target weights evaluated on that training window). (4) The run fails before uploading if `final_optimization` differs from `optimization_metadata.yml` or the `.h5` file's max Sharpe ratio point. The last rebalance still differs from the live portfolio, because it is one holding period older. Date folders written before v38 keep their values and do not have the new keys, and comparisons with them mix training window lengths by one return. Frontend: no change required; treat the new keys as optional; to show the live portfolio next to the backtest, use `final_optimization`, not `rebalances[-1]`; show `in_sample` values only as estimates. |
| 37 | 2026-10-31 | Documentation only; no change to files, keys, attributes, columns, types, units, or values. "Verification status" now records that `portfolio_backtest_metadata.yml` was checked against a real run without upload; `portfolio_backtest_daily.csv` is still listed as not verified against real output. Frontend: no change required; replace its copy with this file. |
| 36 | 2026-10-31 | **Additive.** Two new files in each date folder of the portfolio optimization folder: `portfolio_backtest_daily.csv` (daily returns in percent and wealth in dollars, for the max Sharpe ratio portfolio and an equal-weight baseline) and `portfolio_backtest_metadata.yml` (cumulative return, CAGR, and maximum drawdown of both, every rebalance with its training window, risk-free rate, target weights, and drifted weights, the settings, and the assumptions). No existing key, column, file, value, or unit changes. Both are uploaded after the `.h5` file and before `optimization_metadata.yml`, which is still last, so the availability rule is unchanged. The shared conventions now say that annualization and the "only daily quantity" apply to the two optimization files, because the backtest files hold daily returns and a compounded CAGR; "Converged optimizations only" likewise covers the optimization files, because the backtest's past optimizations have a documented fallback. A run now also fails before uploading if a close or session is missing from the data the backtest needs. Frontend: no change required; optionally plot the wealth of both portfolios and show the metrics with the suggested wording; treat both files as optional for older dates; read the window lengths from the metadata; label `cagr` as CAGR, not as `annualized_return`; do not present the last rebalance's weights as the live optimization's weights. |
| 35 | 2026-10-31 | Documentation only; no change to files, keys, attributes, columns, types, units, or values. Two `optimization_metadata.yml` notes written before v32 now also cover the maximum return portfolio, matching the backend's behavior since v32: the weight-noise rules apply to `max_return_portfolio.weights`, and a run also fails before uploading if the maximum return optimization does not converge. Frontend: no change required beyond applying the weight-noise rules if it shows `max_return_portfolio.weights`; replace its copy with this file. |
| 34 | 2026-10-26 | Documentation only; no change to files, keys, attributes, columns, types, units, or values. The suggested wording for the optimization plot called the random-portfolios sentence a "caption near the plot". It is now the "caption about the random portfolios", and it may be shown either near the plot or inside a collapsible section attached to the plot (such as an accordion), even one collapsed by default, but it must still be shown. Frontend: no change required; keeping the caption in the plot's accordion is allowed. |
| 33 | 2026-10-25 | Documentation only; no change to files, keys, attributes, columns, types, units, or values. The prose now says **volatility** instead of "risk" for the standard deviation of returns, in the shared conventions, both optimization file sections, the `.h5` layout, and the verification status (earlier changelog entries are left as written). Keys and attributes keep their names for compatibility: `optimum_portfolio.risk`, `min_var_portfolio.risk`, `max_return_portfolio.risk`, `assets.<TICKER>.risk`, and the `.h5` attribute `risk_units` all hold volatility, as the shared conventions now state. The suggested x-axis label is now "Annualized Volatility (σ, %)". Frontend: no data change; use "volatility" in user-facing text and the new x-axis label. |
| 32 | 2026-10-25 | **Additive.** `optimization_metadata.yml` gains four optional sections; no existing key, file, value, unit, or upload order changes, and the `.h5` file is unchanged. They explain the plot. `max_return_portfolio` (return, risk, weights) is the highest-return feasible portfolio, the exact top of the efficient frontier, which the backend now also checks against its closed form before upload. `assets` gives each ticker's own annualized return and risk. `constraints` gives long-only, fully invested, and the weight cap in percent. `random_portfolios` gives the count, the seed, the highest and mean random return next to the equal-weight return, the measured and theoretical spread of their returns, and the share of them within a tolerance of the frontier. Added notes on why few random portfolios come near the frontier, and suggested wording that fills in the share. Also added the backend setting `near_frontier_tolerance`. Frontend: no change required; optionally show the suggested sentence (for v32+ dates only) and a table of `assets`. Treat the four sections as optional. |
| 31 | 2026-10-25 | **BREAKING (geometry) of `portfolio_optimization_plot_data.h5`, approved by the human.** No group, dataset, shape, dtype, unit, or attribute is renamed or changed, and `optimization_metadata.yml` is unchanged apart from floating-point noise. The meaning of three groups changes: (1) `tangency_line`, now called the **capital allocation line (CAL)**, runs from (0, `risk_free_rate`) to **exactly** the max Sharpe ratio point and stops there (its last point equals `max_sharpe_ratio` xs and ys); it used to continue to the largest efficient-frontier risk, which implied leverage. It is still 100 points, evenly spaced in `xs`. The name "capital market line" is retired. (2) `efficient_frontier` target returns now run from the minimum variance return to the **highest feasible return** under the optimizer's constraints, instead of to the highest simulated return. That old end could fall short of the feasible maximum (truncating the frontier) or exceed it (a tail of `NaN`), and it depended on the random draws. The frontier now usually reaches further up and right, and `NaN` points are rare. (3) `simulated_portfolios` are now drawn uniformly (seeded, reproducible) from the **same constraint set as the optimizer**, including the per-ticker weight cap that the old draws ignored; the cloud is illustrative only and does not fill the feasible region. Internally, every coordinate in the file now comes from one backend definition of the problem and one statistics function, and the run fails before upload if the invariants under "One definition behind every point" do not hold. The weight cap, the frontier length, the number of random portfolios, and the seed are now backend settings, with values unchanged. Also added suggested UI wording for the plot; the suggested y-axis label is now "Annualized Arithmetic Expected Return (%)". Frontend must: (a) clip `tangency_line` at `max_sharpe_ratio/xs[0]` for date folders written before v31 (harmless for v31 folders, so clipping every date is simplest); (b) relabel the line "Capital allocation line" and stop calling it the capital market line; (c) not assume the frontier ends at the top of the random cloud, and keep breaking the line at `NaN`; (d) use the suggested title, subtitle, axis labels, legend labels (filling in the risk-free rate from `optimization_metadata.yml`), and caption, without repeating the investment-advice disclaimer; (e) read all array lengths from the file. |
| 30 | 2026-10-24 | Non-breaking; no change to keys, values, types, or units. `optimization_metadata.yml` now has a comment line, `# optimum_portfolio is the maximum Sharpe ratio (tangency) portfolio.`, directly above the `optimum_portfolio` key, because the key's name is ambiguous. YAML parsers ignore comments, so the parsed contents are unchanged; the key is not renamed. Frontend: no change required; replace its copy with this file. Where the UI labels this section, call it the maximum Sharpe ratio portfolio. |
| 29 | 2026-10-24 | Backend safeguard; no change to files, keys, columns, types, units, or the meaning of any value. The backend now checks that the max Sharpe ratio and minimum variance optimizations converge; if either does not, the run fails before uploading, as a missing risk-free rate already does (v14). Before v29 an unconverged optimization's weights could have been uploaded without warning. Efficient-frontier points the optimizer cannot reach are still `NaN`, as before. Added the "Converged optimizations only" note to `optimization_metadata.yml`. Frontend: no change required; replace its copy with this file. |
| 28 | 2026-10-24 | **Additive.** `optimization_metadata.yml` gains `min_var_portfolio.weights`: the minimum variance portfolio's weights in percent, one key per ticker, in the same form (and with the same floating-point noise) as `optimum_portfolio.weights`. The backend already computed these weights but did not upload them. No other key, file, value, unit, or upload order changes; the `.h5` file is unchanged. Added the field, a short example, the note that both weight maps follow the weight-noise rules, and suggested wording for showing both portfolios. Frontend: optionally show the minimum variance weights next to the max Sharpe ratio weights (applying the weight-noise rules to both); treat the key as optional, since dates before v28 lack it. |
| 27 | 2026-10-19 | Documentation only; no change to files, keys, columns, types, units, values, or reader behavior. Corrected three stale passages left from before v21 and v24: "Which dates are available" said `optimization_metadata.yml` is uploaded "last of all three" (there are four files per date; it is still last); "Verification status" listed the upload order without `all_forecasts_metadata.yml`; and it said the forecast description matches the backend's `run()` defaults, a function that no longer exists, so it now says that check applied to the pre-v24 method and the current method is checked by code and unit tests only. Frontend: no change required; replace its copy with this file. |
| 26 | 2026-10-18 | Documentation only; no change to files, keys, columns, types, units, or values. The training window length and the number of scored days are backend settings that can change between runs, so the contract no longer states their current values anywhere outside this changelog. Instead it points readers at the data: `forecast_window.training_window_returns` and `evaluation_window.target_trading_days` in `all_forecasts_metadata.yml`, and the rows of `all_forecasts.csv`. The metadata example shows window-dependent values as placeholders. Reader requirement 2 now also forbids hard-coding the window lengths. The suggested forecast-plot text now fills in the training window length (`{w}`, from `training_window_returns` + 1) instead of a fixed figure, like the metrics wording already does with `{n}`. Also fixes a leftover "same 20 days" in the relative MAE note. Frontend must: fill `{w}` in the forecast-plot text from the metadata (or leave the parenthesis out when the file is absent), and make sure no window length or row count is hard-coded. |
| 25 | 2026-10-18 | **Configuration change**; no key, column, file name, type, or unit is renamed, removed, or changed. The backtest now scores the last **60** trading days instead of 20 (set in the backend's `config/modeling_and_optimization.yml`). `all_forecasts.csv` normally has 61 rows (60 backtest + live) instead of 21, covering about three months, and the backend models the last 312 sessions instead of 272. In `all_forecasts_metadata.yml`, `evaluation_window.target_trading_days` is 60, `n_evaluated` and `n_directional` are normally 60, and the evaluation start dates move about three months back; the metrics are computed over 60 days, so they are less noisy than over 20 but still a modest sample. The training window (252 closes) and the optimization files are unchanged. The contract now states the number of scored days and the row count in terms of `target_trading_days` rather than as a fixed number, so a later change to the setting needs no frontend text change. Date folders written before v25 keep their 20-day metrics and 21 rows unless re-run with `--retrain-arima`. Frontend must: fill the number in the suggested metrics wording from `evaluation_window.target_trading_days` (it said "20 trading days" and "Twenty days" before); check that nothing assumes 21 rows or about one month of backtest on the x-axis. |
| 24 | 2026-10-17 | **Methodology change**; no key, column, file name, format, or unit is renamed, removed, or retyped. Each ARIMA forecast is now trained on the preceding trading year (252 closes, 251 daily log returns) instead of 20 returns, and orders are chosen by the **lowest BIC** of one fit per candidate on that whole window instead of by RMSE and BIC rank on rolling 10-return sub-windows; the winning model makes the forecast. Ties (exactly equal BICs) go to fewer parameters, then lower `p`. The backend now models the last 272 sessions; `all_forecasts.csv` still normally has 21 rows (20 backtest + live) and its columns are unchanged. In `all_forecasts_metadata.yml`, values change but keys do not: `forecast_window.training_window_returns` is 251, `training_window_start_date` is about a year before the date folder, `models.arima.order_selection` and `order_selection_tie_break` describe the new rule, and `order_selection_subwindow_returns` equals the training window (251). The optimization files are unchanged: they use exactly the last 252 sessions (previously all downloaded rows, which could be one more session, so `date_updated.date_from` can move one session later). Date folders uploaded before v24 keep their 20-return forecasts unless re-run with `--retrain-arima`, so comparisons across dates mix the two methods; `training_window_returns` tells them apart. Frontend must: update the forecast-plot text to say the model is refit on the previous trading year (see "Suggested user-facing wording"); show any order-selection text from the metadata rather than hard-coding the old rule. |
| 23 | 2026-10-16 | Methodology fix plus one **additive** key; no key, column, format, or unit is renamed, removed, or changed. ARIMA orders are chosen by the lowest average of the RMSE rank and the BIC rank, and with three candidates equal averages are common. Ties used to go to whichever candidate came first in an internal table whose order depended on which parallel job finished first, so the chosen order (and `T_arima`) could differ between runs on the same data. Ties now go to the lower one-step-ahead RMSE, then fewer parameters (`p + q`), then the lower `p`, so runs are reproducible. `all_forecasts_metadata.yml` gains `models.arima.order_selection_tie_break` describing this rule. `T_arima` and the order columns change only on rows where a tie was broken. Date folders uploaded before v23 keep their forecasts unless re-run with `--retrain-arima`, and their orders on tied rows reflect the old, run-dependent tie-break. Frontend: no change required; displaying the new key is optional, and it is absent in older files. |
| 22 | 2026-10-15 | Documentation only; no change to files, keys, or values. The investment-advice disclaimer is consolidated: the frontend shows one fixed wording ("Not investment advice. Every value shown is a backtest or model output computed from historical prices, and none of it predicts future results or recommends any investment.") once, at the bottom of the UI, and it is no longer repeated in the suggested plot, tooltip, and metrics wording. Frontend: keep the single footer disclaimer with this wording and drop any repeats. |
| 21 | 2026-10-15 | **Additive.** New file `all_forecasts_metadata.yml` next to `all_forecasts.csv` in every date folder: the live forecast's training window and forecast date, and, per ticker, RMSE, MAE and MAPE for ARIMA and naive forecasts over the last 20 trading days with an actual close, `relative_mae` (ARIMA MAE ÷ naive MAE), `skill` (1 − relative MAE), and ARIMA directional accuracy, plus units and a description of both models. All keys are lowercase snake_case, **including ticker keys** (`xbi`, not `XBI`). Upload order is now forecasts CSV, forecasts metadata, `.h5`, optimization metadata (still last); the availability rule is unchanged. To score 20 days, the ARIMA input grows from 40 to 41 trading sessions, so `all_forecasts.csv` normally has 21 rows (20 backtest + live) instead of 20; columns are unchanged. Frontend must: read the metrics from this file (match tickers case-insensitively) instead of computing RMSE itself; treat the file as optional for older dates; use the suggested wording; label `relative_mae` as "relative MAE" or show `skill`, not "MASE". |
| 20 | 2026-10-15 | Bug fix, non-breaking for complete data (no change to keys, column names, or units; with no missing closes the output is unchanged). Two gaps in the price data were mishandled. (1) If a whole session was missing, two rows got the same `pred_date` (the day of the gap) and the session after it had no row, violating "rows are unique by `pred_date`"; training windows are now built on the dates actually present, so `pred_date` is always unique, and the session after a whole missing session has no row. (2) If one ticker's close was missing, its next `T_arima` came out empty by accident (the back-transform multiplied by the missing price); it is now deliberately empty, as are that row's order columns (`T_arima_p`, `T_arima_d`, `T_arima_q`), and `T_naive` still uses the most recent observed close. `T_arima` and the order columns are now documented as possibly empty. Added "Missing data" under "How the forecasts are produced". Frontend: handle empty `T_arima` and empty (or float-typed) order columns on such rows. |
| 19 | 2026-10-15 | **Additive** (no existing column renamed, removed, or changed). `all_forecasts.csv` gains a float column `T_naive` per ticker: the naive (random walk) baseline forecast, equal to the most recent observed close before `pred_date` (normally the previous trading day's close), computed from the same data the row's ARIMA model saw. Present on every row. Each ticker's columns are now ordered `T_arima`, `T_arima_p`, `T_arima_d`, `T_arima_q`, `T_naive`, `T` (order is still not part of the contract). Added the `_naive` suffix rule and a naive-forecast sentence to the suggested wording. No error metrics are uploaded. Frontend: treat `T_naive` as optional (absent before v19); optionally plot it and show the wording. |
| 18 | 2026-10-15 | **Additive** (no existing column renamed, removed, or changed). `all_forecasts.csv` gains an integer column `T_arima_d` per ticker, always `1`, so that `(T_arima_p, T_arima_d, T_arima_q)` is the model's full `(p, d, q)` on log prices: ARIMA(p, 0, q) on log returns is ARIMA(p, 1, q) on log prices. Each ticker's columns are now ordered `T_arima`, `T_arima_p`, `T_arima_d`, `T_arima_q`, `T` (order is still not part of the contract). Added "Suggested user-facing wording" for the forecast tooltip and the text near the plot. Frontend: the exact-suffix rule now includes `_arima_d`; the column is optional (absent before v18); show the wording if displaying model orders. |
| 17 | 2026-10-15 | Methodology fix, non-breaking (no change to keys, columns, formats, or units). Two corrections to how ARIMA orders are chosen: (1) when a model fit failed on a rolling sub-window, the previous sub-window's model was reused to score it (or, on the first sub-window, the run crashed); failed sub-windows are now excluded from the order's RMSE and BIC, and an order whose fits all fail ranks last. (2) The leading missing value left by differencing is dropped, so every sub-window holds 10 actual log returns instead of the first holding 9. Selected orders and `T_arima` values can differ slightly from earlier runs; already-uploaded dates keep their forecasts unless re-run with `--retrain-arima`. Frontend: no change required. |
| 16 | 2026-10-15 | **Additive** (no existing column renamed, removed, or changed). `all_forecasts.csv` gains two integer columns per ticker, `T_arima_p` and `T_arima_q`: the order of the ARIMA(p, 0, q) model on log returns that made each row's forecast, present on every row. Each ticker's columns are now ordered `T_arima`, `T_arima_p`, `T_arima_q`, `T` (order is still not part of the contract). Documented the exact-suffix rule for telling forecast, order, and actual columns apart. Frontend must: check that it identifies forecast columns by the exact `_arima` suffix (not by containing `_arima`) and actual columns by exact ticker match; otherwise the new columns could be misread as forecasts. Displaying the orders is optional. Older date folders lack these columns, so readers must treat them as optional. |
| 15 | 2026-10-15 | Methodology fix, non-breaking (no change to keys, columns, formats, or units; `pred_date` was already the next NYSE trading day). ARIMA training windows were resampled to pandas weekdays (`"B"`), which inserted NYSE holidays such as Presidents Day as missing observations inside the window. Windows now contain only NYSE trading sessions, so a model time step is always one session. `T_arima` values for windows that span an NYSE holiday can differ slightly from earlier runs; date folders already uploaded keep their old forecasts unless the backend is re-run with `--retrain-arima`. Frontend: no change required. |
| 14 | 2026-10-14 | Bug fix, non-breaking (no change to keys, formats, or units). FRED leaves `DTB3` empty on bond-market holidays (e.g. Columbus Day); the backend used the latest row even when it was empty, which could make `risk_free_rate`, `daily_risk_free_rate`, `sharpe_ratio`, and the tangency line `NaN`. It now uses the most recent non-missing value, so `risk_free_rate_date` can be earlier than the last FRED row, and a run with no valid rate fails before uploading. Clarified the two `risk_free_rate` field descriptions. Frontend: no change required. |
| 13 | 2026-10-12 | **Configuration rename; no objects move and key strings are unchanged.** The environment variables naming the two top-level folders are renamed so their names say what they hold: `PORTFOLIO_OPTIMIZATION_SPACE_NAME` is now `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`, and `TIME_SERIES_SPACE_NAME` is now `FSF_FRONT_END_TIME_SERIES_FOLDER`. Values are unchanged. No fallback: the backend reads only the new names, and the human has already renamed them in the backend `.env` for dev and production. Documented that dev and production share the one Space and differ only in folder names, and listed `FSF_FRONT_END_BUCKET_NAME` with the folder variables. Frontend must: read the new names, rename its Hugging Face secrets and any local `.env` (same values), and drop the old names. |
| 12 | 2026-10-12 | Documentation only; no change to keys, formats, or reader behavior. The v11 text was amended in place after it was first written, so a copy taken earlier may differ. This version is the reconciled text for both repos. Amendments since the first v11 text: the "Connecting" section states that the backend also reads `FSF_FRONT_END_BUCKET_NAME` and uploads with `Bucket=<space>`, `Key="<folder>/YYYY-MM-DD/<file>"`; "Verification status" records the programmatic listing of the real keys, the deleted stale object, and a second listing after the backend change; the v11 changelog entry notes the backend change. Frontend: replace its copy with this file. |
| 11 | 2026-10-12 | **Documentation correction; no objects move and key strings are unchanged.** Earlier versions described `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME` as bucket names. They are top-level folder names inside **one** Space (bucket), so the full keys are `<folder>/YYYY-MM-DD/<file>`. Uploads landed correctly only because boto3, given an endpoint containing the Space name, sends path-style requests. Rewrote "Where the data lives", "The date prefix", "Which dates are available", the per-file Folder/Key lines, and reader requirement 7; added "Connecting". Frontend must: add `FSF_FRONT_END_BUCKET_NAME` (the Space name), use the region-only endpoint `https://<region>.digitaloceanspaces.com`, list each folder with `ListObjectsV2(Bucket=<space>, Prefix="<folder>/")` with pagination, strip the folder prefix before matching dates, and read objects with `Bucket=<space>`, `Key="<folder>/YYYY-MM-DD/<file>"`. The backend was changed the same way (it reads `FSF_FRONT_END_BUCKET_NAME` and passes the folder in the key), which produces byte-identical object keys. |
| 10 | 2026-10-12 | Non-breaking; no change to keys, file names, formats, columns, shapes, or units. The backend now has a single entry point that produces all three files in one run: one price download, ARIMA trained on the last 40 trading sessions of that data (the same window as before), then the optimization. Uploads happen only after every computation succeeds, in the order `all_forecasts.csv`, `.h5`, `optimization_metadata.yml`, so the metadata file now marks a complete date across both buckets. The availability rule (both buckets) is unchanged because earlier dates came from two separate scripts. |
| 9 | 2026-10-12 | **Breaking (keys).** Every file moves under a date prefix `YYYY-MM-DD/` (the data as-of date = last NYSE trading day in the run's data), keeping one folder per run instead of overwriting. File names, formats, columns, shapes, and units are unchanged. Root-level keys are **no longer written** (clean break, no dual-write); stale root objects remain until removed by hand. Defined date availability as the intersection of both buckets, with `optimization_metadata.yml` as the optimization completion marker. Frontend must: list date prefixes in both buckets (`ListObjectsV2`, `Delimiter="/"`, paginated), keep only dates complete in both, show them in a dropdown newest-first with the newest selected by default, load all files from the selected date's folder, and ignore root-level objects. Ship the frontend change with the first prefixed upload to production. |
| 8 | 2026-10-12 | Calendar fix, non-breaking (no change to keys, columns, shapes, or units): business days now follow the NYSE trading calendar instead of the US federal holiday calendar. Effects: `pred_date` and the ARIMA training windows skip NYSE closures (e.g. Good Friday) and include federal-only holidays (e.g. Columbus Day, Veterans Day), and downloaded data now ends on the last NYSE session instead of possibly omitting one. |
| 7 | 2026-10-11 | Documentation only: recorded verification of the `.h5` point values and tangency line against real output. No change to keys, shapes, units, or meaning. |
| 6 | 2026-10-11 | Documentation only: replaced the illustrative v4 YAML example with real output and recorded the verification of the v4 YAML keys and `.h5` root attributes. No change to keys, shapes, units, or meaning. |
| 5 | 2026-10-11 | Bug fix, non-breaking: the tangency line's risk range now ends at the largest **non-NaN** efficient-frontier risk. Before, a `NaN` frontier point could make the whole `tangency_line` `NaN`. Documented that `tangency_line` contains no `NaN`. No change to keys, shapes, or units. |
| 4 | 2026-10-11 | **Breaking (units).** Every return and risk in the optimization files is now annualized arithmetically (return × 252, risk × √252). `.h5`: every dataset changes from daily % to annualized %, with the same keys and shapes, plus new root attributes describing units. YAML: `optimum_portfolio.annualized_return` changes from compounded to arithmetic (about 40.9 → 34.3 on the 2026-10-09 data). Additive: `optimum_portfolio.sharpe_ratio`, `min_var_portfolio`, `annualization`. `daily_risk_free_rate` kept unchanged. Frontend must: relabel axes as annualized, stop any daily→annual conversion it does, check the `.h5` `annualization` attribute (absent means a pre-v4 daily file), and update any display of `annualized_return`. |
| 3 | 2026-10-11 | Backend no longer renders a local PNG plot (it was never uploaded). Removed the note about it. No change to uploaded keys, formats, shapes, or units. |
| 2 | 2026-10-11 | Replaced the illustrative YAML example with a real file. Documented floating-point noise in weights (scientific notation, `~1e-14` for zero, `39.99999999999994` at the cap) and reader handling. Recorded verification status. No change to keys, columns, shapes, or units. |
| 1 | 2026-10-11 | Initial contract, derived from the backend code. |
