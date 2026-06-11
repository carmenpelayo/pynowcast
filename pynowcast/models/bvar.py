import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.linear_model import Ridge


class BayesianVARModel:
    """
    Bayesian VAR model wrapper. Uses statsmodels VAR for OLS estimation
    with optional ridge regularization as a Bayesian shrinkage analogue.

    Attributes:
        endog: pandas DataFrame of endogenous variables.
        lags: number of lags.
        use_ridge: whether to apply ridge regularization.
        alpha: ridge penalty parameter.
        model: statsmodels VAR instance.
        results: fitted results or custom coefficients.
    """
    def __init__(self,
                 endog: pd.DataFrame,
                 lags: int = 1,
                 use_ridge: bool = False,
                 alpha: float = 1.0):
        self.endog = endog
        self.lags = lags
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.model = None
        self.results = None

    def fit(self, **fit_kwargs) -> None:
        """
        Fit the VAR model. If use_ridge is False, use statsmodels VAR.
        Otherwise, fit each equation with Ridge regression.

        Args:
            **fit_kwargs: passed to statsmodels VAR.fit().
        """
        if not self.use_ridge:
            self.model = VAR(self.endog)
            self.results = self.model.fit(self.lags, **fit_kwargs)
        else:
            # Prepare lagged design matrix
            data = self.endog
            lagged = self.model = VAR(data).fit(maxlags=self.lags).prepare_data(
                dynamic=False)
            # X: lagged exog, y: current values
            X = lagged[0]
            y = lagged[1]
            coefs = {}
            for i, col in enumerate(data.columns):
                ridge = Ridge(alpha=self.alpha)
                ridge.fit(X, y[:, i])
                coefs[col] = ridge.coef_
            self.results = coefs

    def nowcast(self, steps: int = 1) -> pd.DataFrame:
        """
        Generate nowcasts for the next `steps` periods.

        Args:
            steps: number of ahead periods to forecast.

        Returns:
            DataFrame of forecasts.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting.")
        if not self.use_ridge:
            return self.results.forecast(self.endog.values[-self.lags:], steps)
        else:
            # Ridge-based manual forecasting not implemented
            raise NotImplementedError("Forecasting with ridge prior not implemented.")

    def summary(self) -> None:
        """
        Print summary of model.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before summarizing.")
        if not self.use_ridge:
            print(self.results.summary())
        else:
            print("Ridge VAR coefficients:")
            for var, coef in self.results.items():
                print(f"Variable {var}: Coefs shape {coef.shape}")
