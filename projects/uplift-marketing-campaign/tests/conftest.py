import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic():
    """Randomised experiment with a known, sign-changing treatment effect.

    Baseline response 0.30. Customers with x0 > 0 are persuadables (+0.25),
    customers with x0 <= 0 are sleeping dogs (-0.15). x1..x3 are noise/baseline drivers.
    """
    rng = np.random.default_rng(0)
    n = 8000
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, 4)), columns=[f"x{i}" for i in range(4)])
    tau = np.where(X["x0"] > 0, 0.25, -0.15)
    treatment = rng.integers(0, 2, n)
    p = 0.30 + 0.1 * X["x1"].to_numpy() + tau * treatment
    y = rng.binomial(1, p)
    return X, treatment, y, tau
