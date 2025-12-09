import os
import numpy as np
import pandas as pd
from data_loader import load_price_data, compute_log_returns, estimate_moments
from simulator import MonteCarloPortfolioSimulator
from risk_metrics import compute_var, compute_cvar
from stress_tests import historical_stress_test, scenario_stress_test

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "prices_sample.csv")
    prices = load_price_data(data_path)
    print("Loaded prices for assets:", list(prices.columns))

    returns = compute_log_returns(prices)
    mu, cov, asset_names = estimate_moments(returns)
    n_assets = len(asset_names)
    weights = np.ones(n_assets) / n_assets

    print("\nEstimated daily mean returns:")
    for name, m in zip(asset_names, mu):
        print(f"{name:10s}: {m: .4%}")

    simulator = MonteCarloPortfolioSimulator(mu, cov, weights)
    n_scenarios = 20000
    simulated_portfolio_returns = simulator.simulate_portfolio_returns(n_scenarios=n_scenarios, horizon_days=1)

    for alpha in [0.95, 0.99]:
        var = compute_var(simulated_portfolio_returns, alpha=alpha)
        cvar = compute_cvar(simulated_portfolio_returns, alpha=alpha)
        print(f"\n=== Risk Metrics (alpha={alpha:.2f}) ===")
        print(f"{int(alpha*100)}% VaR : {var: .2%}")
        print(f"{int(alpha*100)}% CVaR: {cvar: .2%}")

    # Historical stress example (ensure dates exist in sample)
    try:
        stress_start = returns.index.min().strftime('%Y-%m-%d')
        stress_end = (returns.index.min() + pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        stress_period_returns = historical_stress_test(returns, weights, stress_start, stress_end)
        print("\n=== Historical Stress Test Example ===")
        print("Min daily return in window : ", f"{stress_period_returns.min(): .2%}")
        print("Cumulative return over window:", f"{(1+stress_period_returns).prod()-1: .2%}")
    except Exception as e:
        print("Historical stress test skipped:", e)

    # Scenario stress test: all assets -25%
    crash = np.full(n_assets, -0.25)
    scenario_stress_test(weights, crash, asset_names=asset_names, scenario_name='-25% crash (all assets)')

if __name__ == '__main__':
    main()
