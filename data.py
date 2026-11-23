"""Data with a known answer so explanation methods can be graded.

y = 3*x0 + 2*x1^2 + 4*x2*x3 + noise, with x4 a near-copy of x0 and x5..x7 pure noise.
    x0  linear effect
    x1  non-linear (U-shaped) effect
    x2, x3  only matter together (interaction)
    x4  correlated with x0 but has no effect of its own
    x5-x7  noise
"""
import numpy as np
import pandas as pd

NAMES = ["x0_linear", "x1_quadratic", "x2_interact", "x3_interact", "x4_copy_of_x0", "x5_noise", "x6_noise", "x7_noise"]


def make(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 8))
    X[:, 4] = X[:, 0] + rng.normal(0, 0.1, n)
    y = 3 * X[:, 0] + 2 * X[:, 1] ** 2 + 4 * X[:, 2] * X[:, 3] + rng.normal(0, 0.2, n)
    return pd.DataFrame(X, columns=NAMES), y


def true_contributions(X: pd.DataFrame) -> np.ndarray:
    """Exact Shapley values of the true function (independent uniform features, x4 has none)."""
    x = X.to_numpy()
    out = np.zeros_like(x)
    out[:, 0] = 3 * x[:, 0]                       # mean of x0 is 0
    out[:, 1] = 2 * (x[:, 1] ** 2 - 1 / 3)        # E[x1^2] = 1/3 for U(-1, 1)
    out[:, 2] = out[:, 3] = 2 * x[:, 2] * x[:, 3]  # interaction split evenly between the pair
    return out
