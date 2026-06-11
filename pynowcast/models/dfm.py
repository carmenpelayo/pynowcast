import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


class DynamicFactorModel:
    """
    Dynamic Factor Model wrapper around statsmodels DynamicFactor.

    Attributes:
        endog: Observed data (pandas DataFrame, columns as series).
        k_factors: Number of common factors.
        factor_order: AR order of factors.
        error_order: AR order of idiosyncratic errors.
        model: statsmodels DynamicFactor instance.
        results: Fitted model results.
    """
    def __init__(self,
                 endog: pd.DataFrame,
                 k_factors: int,
                 factor_order: int = 1,
                 error_order: int = 1):
        """
        Initialize the Dynamic Factor Model.

        Args:
            endog: DataFrame with observed series (columns).
            k_factors: Number of latent common factors.
            factor_order: AR order for the factors.
            error_order: AR order for idiosyncratic errors.
        """
        self.endog = endog
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.model = None
        self.results = None

    def fit(self, **fit_kwargs) -> None:
        """
        Fit the Dynamic Factor Model to the data.

        Args:
            **fit_kwargs: Keyword arguments passed to statsmodels DynamicFactor.fit().
        """
        self.model = DynamicFactor(
            endog=self.endog,
            k_factors=self.k_factors,
            factor_order=self.factor_order,
            error_order=self.error_order
        )
        self.results = self.model.fit(**fit_kwargs)

    def nowcast(self) -> pd.DataFrame:
        """
        Generate nowcasts for the endogenous variables.

        Returns:
            DataFrame of predicted (nowcasted) values indexed like endog.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before creating nowcasts.")
        return self.results.predict()

    def summarize(self) -> None:
        """
        Print a summary of the fitted model.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before summarizing.")
        print(self.results.summary())
