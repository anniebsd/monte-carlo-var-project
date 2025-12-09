import numpy as np
import pandas as pd

def historical_stress_test(returns_df: pd.DataFrame,
                           weights: np.ndarray,
                           start_date: str,
                           end_date: str) -> pd.Series:
    window = returns_df.loc[start_date:end_date]
    portfolio_returns = window.values @ weights
    return pd.Series(portfolio_returns, index=window.index, name="portfolio_return")

def scenario_stress_test(weights: np.ndarray,
                         scenario_returns: np.ndarray,
                         asset_names=None,
                         scenario_name: str = "Custom Scenario") -> float:
    weights = np.asarray(weights)
    scenario_returns = np.asarray(scenario_returns)
    if weights.shape[0] != scenario_returns.shape[0]:
        raise ValueError("weights and scenario_returns must have same length")
    portfolio_return = float(weights @ scenario_returns)
    print(f"\n=== Scenario Stress Test: {scenario_name} ===")
    if asset_names is not None:
        for name, r in zip(asset_names, scenario_returns):
            print(f"{name:15s}: {r: .2%}")
    print(f"Portfolio return: {portfolio_return: .2%}")
    return portfolio_return
