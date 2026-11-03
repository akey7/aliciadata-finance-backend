# Fuzzy System Research and Finance: Methods and Results

## 1. Project Overview

This project investigates two related quantitative problems: (1) how historical returns can be used to construct portfolios along the mean-variance efficient frontier, and (2) whether short-horizon ARIMA forecasts of individual securities outperform a simple persistence benchmark under walk-forward evaluation.

In contrast to my other projects focusing on life sciences data and renewable energy, this project aims to showcase a project where (1) I am the sole contributor and (2) since the data source is public and well-known, domain expertise of neither renewable energy nor biology are required to understand what is going on.

If you want to jump straight to portfolio optimization or ARIMA time series modeling, you can select one of the following links:

- [2. Portfolio Optimization](#2-portfolio-optimization)
- [3. Time-Series Forecasting](#3-time-series-forecasting)

### 1.1 Objectives

The purpose of this project is to provide a portfolio piece showcasing two problems focused on time series data. First and is solving optimization training second is training a machine learning model (ARIMA, auto-regressive integrated moving average) on the time series data. The dataset is publicly available financial market data. Also demonstrated is thourough attention to testing model performance with out-of-sample backtests that avoid temporal leakage (inadvertantly using future data to evaluate past performance).

### 1.2 System Architecture

The project is split into 2 repos, one each for the backend and frontend. The sepration of concerns is simple: the backend downloads data, trains models and runs optimizations, and uploads the data to the frontend for display. The frontend simply displays the data calculated by the backend. The code for the backend is more complicated and takes advantage of multicore hardware to accelerate computations. In contrast, the frontend has less code, fewer dependencies, and is made to deploy to low-cost/low-resource hardware. An interface contract defines the files and schemas of data in the S3 bucket transferred from the backend to the frontend.

In turn, this backend is split into subpackages devoted to downloading market data, running portoflio optimizations, training ARIMA models for forecasting, and uploading daily data to an S3 bucket for the frontend to display.

System architecture is not the focus of this document. Interested readers are pointed to the 

### 1.3 Data Sources

Stock market data is downloaded from the [Massive.com](https://massive.com/) API. Risk-free rate 3-month T-Bill data is downloaded from FRED (Federal Reserve Economic Data, maintained by the Federal Reserve Bank of St. Louis), series DTB3.

### 1.4 Reproducibility and Software Environment

## 2. Portfolio Optimization
### 2.1 Introduction
### 2.2 Methods

**Equation 1.** Expected portfolio return

```math
E[R_p] = \mathbf{w}^{T}\boldsymbol{\mu}
```

**Equation 2.** Portfolio variance

```math
\sigma_p^2 = \mathbf{w}^{T}\Sigma\mathbf{w}
```

- constraints on **w**;
- minimum-variance portfolio;
- maximum-Sharpe portfolio;
- risk-free rate;
- efficient-frontier computation;
- annualization;
- solver/numerical method;
- Monte Carlo simulation.

### 2.3 Results

- What does the efficient frontier look like for the current asset universe?
- How different are the max-Sharpe and minimum-variance allocations?
- Where do Monte Carlo portfolios fall relative to the numerical frontier?

### 2.4 Discussion

## 3. Time-Series Forecasting
### 3.1 Introduction
### 3.2 Methods
### 3.3 Results
### 3.4 Discussion

## 4. Integration and Future Work

The optimization and forecasting components are currently evaluated independently. Forecast estimates are not used as expected-return inputs to portfolio optimization because the forecasting analysis does not demonstrate sufficient out-of-sample improvement over a naive persistence benchmark. Integration therefore remains a potential extension rather than an assumption of the current system.

