# Fuzzy System Research

**Quantitative finance backend for portfolio optimization and walk-forward financial time-series forecasting.**

Fuzzy System Research is the analytical backend for the [Fuzzy System Finance Dashboard](https://huggingface.co/spaces/akey7/fuzzy-system-finance), an interactive dashboard for exploring portfolio construction and forecast performance using public financial-market data.

The project implements two quantitative workflows:

- **Portfolio optimization:** constrained mean-variance optimization, minimum-variance and maximum-Sharpe portfolios, efficient-frontier construction, and Monte Carlo portfolio simulation, with a walk-forward out-of-sample backtest of the maximum-Sharpe portfolio against an equal-weight baseline.
- **Time-series forecasting:** ARIMA models fit to rolling historical windows and evaluated using walk-forward out-of-sample forecasts against a naive benchmark. Forecast performance is quantified with RMSE, MAE, MAPE, relative MAE, forecast skill, and directional accuracy.

The backend retrieves market data, performs modeling and validation, records reproducible dated outputs, and publishes completed results to object storage consumed by a separately deployed frontend. Model configuration is explicitly validated, computationally expensive forecasts are cached, independent ARIMA fits can execute in parallel, and results are uploaded only after the analytical pipeline succeeds.

This project is intended both as a quantitative-methods demonstration and as an example of turning statistical analysis into a reproducible analytical system.

## Documentation

**Quantitative methods and interpretation:** [docs/METHODS_AND_RESULTS.md](docs/METHODS_AND_RESULTS.md)

**System Design:** [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)

**Interface Contract Between Backend and Frontend:** [docs/INTERFACE_CONTRACT.md](docs/INTERFACE_CONTRACT.md)

**Installation, execution, and testing:** [docs/INSTALLATION_AND_EXECUTION.md](docs/INSTALLATION_AND_EXECUTION.md)

## Related Links

- **Live dashboard:** [Fuzzy System Finance](https://huggingface.co/spaces/akey7/fuzzy-system-finance)
- **Frontend repository:** [akey7/fuzzy-system-finance](https://github.com/akey7/fuzzy-system-finance)
