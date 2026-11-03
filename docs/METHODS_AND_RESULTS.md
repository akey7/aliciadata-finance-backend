# Fuzzy System Research and Finance: Methods and Results

## 1. Project Overview

This project investigates two related quantitative problems: (1) how historical returns can be used to construct portfolios along the mean-variance efficient frontier, and (2) whether short-horizon ARIMA forecasts of individual securities outperform a simple persistence benchmark under walk-forward evaluation.

### 1.1 Objectives
### 1.2 System Architecture
### 1.3 Data Sources and Shared Preprocessing
### 1.4 Reproducibility and Software Environment

## 2. Portfolio Optimization
### 2.1 Introduction
### 2.2 Methods

Equation 1. Expected portfolio return
$$ E[R_p] = \mathbf{w}^{T}\boldsymbol{\mu} $$

Equation 2. Portfolio variance
$$ \sigma_p^2 = \mathbf{w}^{T}\Sigma\mathbf{w} $$

- constraints on **w**;
- minimum-variance portfolio;
- maximum-Sharpe portfolio;
- risk-free rate;
- efficient-frontier computation;
- annualization;
- solver/numerical method;
- Monte Carlo simulation.
- What does the efficient frontier look like for the current asset universe?
- How different are the max-Sharpe and minimum-variance allocations?
- Where do Monte Carlo portfolios fall relative to the numerical frontier?

### 2.3 Results
### 2.4 Discussion

## 3. Time-Series Forecasting
### 3.1 Introduction
### 3.2 Methods
### 3.3 Results
### 3.4 Discussion

## 4. Integration and Future Work
