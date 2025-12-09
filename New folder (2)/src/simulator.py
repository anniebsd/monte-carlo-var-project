import numpy as np

class MonteCarloPortfolioSimulator:
    def __init__(self, mu, cov, weights):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.weights = np.asarray(weights)
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array")
        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError("cov must be square")
        if self.cov.shape[0] != self.mu.shape[0]:
            raise ValueError("mu and cov dimensions must match")
        if self.weights.shape[0] != self.mu.shape[0]:
            raise ValueError("weights dimension must match number of assets")

    def simulate_asset_returns(self, n_scenarios: int, horizon_days: int = 1) -> np.ndarray:
        n_assets = self.mu.shape[0]
        mu_h = self.mu * horizon_days
        cov_h = self.cov * horizon_days
        # ensure positive-definite (small jitter)
        jitter = 1e-10
        cov_h = cov_h + np.eye(n_assets)*jitter
        L = np.linalg.cholesky(cov_h)
        Z = np.random.randn(n_scenarios, n_assets)
        correlated = Z @ L.T
        simulated_returns = correlated + mu_h
        return simulated_returns

    def simulate_portfolio_returns(self, n_scenarios: int, horizon_days: int = 1) -> np.ndarray:
        asset_returns = self.simulate_asset_returns(n_scenarios, horizon_days)
        portfolio_returns = asset_returns @ self.weights
        return portfolio_returns
