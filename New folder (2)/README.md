# Monte Carlo VaR, CVaR & Stress Testing for a Multi-Asset Portfolio

This repo contains a polished Monte Carlo risk engine implemented in Python.

## What's included
- `src/` : core modules
  - `data_loader.py` : load prices & compute returns
  - `simulator.py` : Monte Carlo simulation using Cholesky decomposition
  - `risk_metrics.py` : VaR & CVaR calculations
  - `stress_tests.py` : historical & scenario stress tests
  - `main.py` : example end-to-end run using the sample data
- `data/prices_sample.csv` : simulated sample price data for 4 tickers (AAPL, MSFT, GOOG, AMZN)
- `notebooks/MonteCarlo_VaR_CVaR_Stress.ipynb` : Jupyter notebook with explanations and runnable cells

## Quick start
```bash
git clone <this-repo>
cd monte_carlo_var_project
pip install numpy pandas matplotlib
python src/main.py
```

The project is ready to upload to GitHub. Replace `data/prices_sample.csv` with real market data or enable Yahoo Finance in the notebook to download live data.
