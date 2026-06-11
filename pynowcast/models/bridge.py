import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


class BridgeRegression:
    """
    Bridge regression model: fits a regression of a target series
    on a set of predictor series (e.g., factors from a DFM).

    Attributes:
        model: sklearn regression model instance.
        method: 'ols' or 'ridge'.
        alpha: regularization strength for ridge.
        features: feature column names.
        target: name of the target series.
    """
    def __init__(self,
                 method: str = 'ols',
                 alpha: float = 1.0):
        self.method = method.lower()
        self.alpha = alpha
        self.model = None
        self.features = None
        self.target = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the bridge regression.

        Args:
            X: DataFrame of predictors.
            y: Series of target.
        """
        self.features = X.columns.tolist()
        self.target = y.name
        if self.method == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for given predictors.

        Args:
            X: DataFrame of predictors.

        Returns:
            Array of predicted values.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def summary(self) -> None:
        """
        Print summary of fitted bridge regression.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before summarizing.")
        coef = self.model.coef_
        intercept = self.model.intercept_
        print(f"Bridge Regression ({self.method.upper()}):")
        print(f"Intercept: {intercept}")
        for feat, c in zip(self.features, coef):
            print(f"{feat}: {c}")
