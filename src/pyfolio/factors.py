"""Linear factor models for asset pricing and risk analysis.

This module implements linear factor models (CAPM, Fama-French, Carhart)
to decompose asset returns into alpha, beta, and factor exposures.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TRADING_DAYS = 252


class LinearFactorModel:
    """
    Linear factor model:

        R_i - RF = alpha_i + sum_k beta_{ik} * F_k + eps_i

    Produces:
    - alpha_i
    - beta_{ik}
    - residual variances sigma_ei^2
    - expected returns (historical factor means)
    - factor-implied covariance matrix

    Works for:
    - SIM        : factor_cols = ["Mkt-RF"]
    - CAPM       : factor_cols = ["Mkt-RF"]
    - FF3        : factor_cols = ["Mkt-RF", "SMB", "HML"]
    - Carhart    : add "MOM"
    - FF5        : add "RMW", "CMA"
    """

    def __init__(self, rf_col="RF"):
        self.rf_col = rf_col
        self.alpha = None  # (N,)
        self.betas = None  # (N, K)
        self.resid_var = None  # (N,)
        self.assets = None
        self.factor_mean = None  # (K,)
        self.factor_cov = None  # (K, K)
        self.rf_mean = None  # scalar

    # ------------------------------------------------------------
    # 1) Fit factor model
    # ------------------------------------------------------------
    def fit(self, asset_returns: pd.DataFrame, factors: pd.DataFrame):
        """
        asset_returns : T x N DataFrame (decimal returns)
        factors       : T x (K + RF) DataFrame (decimal returns)
        """
        df = asset_returns.join(factors, how="inner").dropna()
        self.assets = asset_returns.columns.tolist()

        self.factor_cols = factors.columns.difference([self.rf_col]).tolist()

        R = df[self.assets].values  # T x N
        RF = df[self.rf_col].values.reshape(-1, 1)  # T x 1
        Y = R - RF  # excess returns

        X = df[self.factor_cols].values  # T x K

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, Y)

        self.alpha = reg.intercept_  # (N,)
        self.betas = reg.coef_  # (N, K)

        # residual variance (regression-correct)
        Y_hat = reg.predict(X)
        eps = Y - Y_hat
        T, K = X.shape
        self.resid_var = (eps**2).sum(axis=0) / (T - (K + 1))

        # factor moments (historical)
        self.factor_mean = X.mean(axis=0)  # E[F]
        self.factor_cov = np.atleast_2d(np.cov(X, rowvar=False))
        self.rf_mean = RF.mean().item()  # E[RF]

        return self

    # ------------------------------------------------------------
    # 2) Expected returns μ (lecture-consistent)
    # ------------------------------------------------------------
    def expected_returns(self, annualize=True):
        """
        E[R_i] = alpha_i + betas_i * E[F] + E[RF]
        """
        mu_excess = self.alpha + self.betas @ self.factor_mean
        mu = mu_excess + self.rf_mean

        if annualize:
            mu = mu * TRADING_DAYS

        return pd.Series(mu, index=self.assets, name="mu")

    # ------------------------------------------------------------
    # 3) Covariance matrix Σ
    # ------------------------------------------------------------
    def covariance(self, annualize=True):
        """
        Σ = B Σ_F B' + D
        """
        Sigma = self.betas @ self.factor_cov @ self.betas.T + np.diag(self.resid_var)

        if annualize:
            Sigma = Sigma * TRADING_DAYS

        return pd.DataFrame(Sigma, index=self.assets, columns=self.assets)
