import numpy as np
import pandas as pd


def mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error"""
    # avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def theils_u(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Theil's U statistic"""
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2))
    if denom == 0:
        return np.nan
    return num / denom


def evaluate_forecasts(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa múltiples series de pronóstico.

    Args:
        y_true: DataFrame de observaciones reales (columnas = series).
        y_pred: DataFrame de pronósticos (columnas deben coincidir).

    Returns:
        DataFrame con métricas para cada serie: ['MSE', 'RMSE', 'MAE', 'MAPE', 'Theil_U'].
    """
    metrics = []
    for col in y_true.columns:
        true = y_true[col].dropna()
        pred = y_pred[col].reindex_like(true)
        metrics.append({
            'series': col,
            'MSE': mean_squared_error(true, pred),
            'RMSE': root_mean_squared_error(true, pred),
            'MAE': mean_absolute_error(true, pred),
            'MAPE': mean_absolute_percentage_error(true, pred),
            'Theil_U': theils_u(true, pred)
        })
    return pd.DataFrame(metrics).set_index('series')
