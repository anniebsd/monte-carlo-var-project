import pandas as pd
import numpy as np
from pathlib import Path

def load_price_data(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    log_prices = np.log(price_df)
    log_returns = log_prices.diff().dropna()
    return log_returns

def estimate_moments(returns_df: pd.DataFrame):
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    asset_names = list(returns_df.columns)
    return mu, cov, asset_names
