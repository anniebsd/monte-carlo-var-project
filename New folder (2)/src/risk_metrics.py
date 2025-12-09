import numpy as np

def compute_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    returns = np.sort(returns)
    n = len(returns)
    idx = int((1 - alpha) * n)
    idx = max(0, min(idx, n-1))
    var_level_return = returns[idx]
    var = -var_level_return
    return float(var)

def compute_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    returns = np.sort(returns)
    n = len(returns)
    idx = int((1 - alpha) * n)
    idx = max(0, min(idx, n-1))
    tail = returns[: idx + 1]
    cvar = -tail.mean()
    return float(cvar)
