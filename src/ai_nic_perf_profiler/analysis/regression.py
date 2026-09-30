"""Non-negative ridge regression for additive step-time decomposition.

We model excess step time as ``y = X @ w`` with ``w >= 0``: a network anomaly
can only add time, never remove it. The non-negativity constraint also keeps
correlated features (ECN marks and CNPs rise together) from getting opposite
signs that cancel out and mean nothing. Ridge regularisation splits credit
between collinear features instead of arbitrarily giving it all to one.

For a linear model with a baseline reference point, the per-feature
contribution ``w_f * (x_f - x_f_baseline)`` is exactly that feature's Shapley
value, so the decomposition is additive and order-independent without any
sampling.
"""

from __future__ import annotations

import numpy as np


def nnls_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0, max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Solve ``min ||Xw - y||^2 + lam ||w||^2  s.t. w >= 0`` by projected coordinate descent."""
    n, k = X.shape
    w = np.zeros(k)
    if n == 0 or k == 0:
        return w
    gram = X.T @ X + lam * np.eye(k)
    xty = X.T @ y
    for _ in range(max_iter):
        max_step = 0.0
        for j in range(k):
            # Gradient of the objective w.r.t. w_j with w_j excluded, then project.
            resid = xty[j] - gram[j] @ w + gram[j, j] * w[j]
            new = max(0.0, resid / gram[j, j])
            max_step = max(max_step, abs(new - w[j]))
            w[j] = new
        if max_step < tol:
            break
    return w


def r_squared(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - float(((y - X @ w) ** 2).sum()) / ss_tot
