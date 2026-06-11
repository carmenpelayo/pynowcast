import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV


def select_by_correlation(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.Index:
    """
    Selecciona las k variables con mayor correlación absoluta respecto al target.

    Args:
        X: DataFrame de predictores.
        y: Serie target.
        k: Número de variables a seleccionar.

    Returns:
        Índice de columnas seleccionadas.
    """
    corrs = X.apply(lambda col: np.abs(np.corrcoef(col.fillna(col.mean()), y.fillna(y.mean()))[0, 1]))
    return corrs.sort_values(ascending=False).head(k).index


def select_by_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 10, random_state: int = 0) -> pd.Index:
    """
    Selecciona las k variables con mayor información mutua respecto al target.

    Args:
        X: DataFrame de predictores.
        y: Serie target.
        k: Número de variables a seleccionar.
        random_state: Semilla para reproducibilidad.

    Returns:
        Índice de columnas seleccionadas.
    """
    # mutual_info_regression requiere valores finitos
    X_filled = X.fillna(X.mean())
    y_filled = y.fillna(y.mean())
    mi = mutual_info_regression(X_filled, y_filled, random_state=random_state)
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series.sort_values(ascending=False).head(k).index


def select_by_lasso(X: pd.DataFrame, y: pd.Series, cv: int = 5, alpha_min: float = None, max_iter: int = 10000) -> pd.Index:
    """
    Selecciona variables usando LASSO con validación cruzada.

    Args:
        X: DataFrame de predictores.
        y: Serie target.
        cv: Número de folds para validación cruzada.
        alpha_min: Valor mínimo de alpha para la ruta de LASSO; si None, usa valores por defecto.
        max_iter: Iteraciones máximas para el solver.

    Returns:
        Índice de columnas seleccionadas (coef != 0).
    """
    X_filled = X.fillna(X.mean())
    y_filled = y.fillna(y.mean())
    lasso = LassoCV(cv=cv, n_alphas=100 if alpha_min is None else None,
                    alphas=None if alpha_min is None else np.logspace(np.log10(alpha_min), 0, 100),
                    max_iter=max_iter).fit(X_filled, y_filled)
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected = coef[coef.abs() > 1e-6].index
    return selected


def rank_variables(X: pd.DataFrame, y: pd.Series, methods: list = None, k: int = 10) -> pd.DataFrame:
    """
    Genera un ranking de variables combinando múltiples métodos de selección.

    Args:
        X: DataFrame de predictores.
        y: Serie target.
        methods: Lista de métodos a aplicar: 'corr', 'mi', 'lasso'. Si None, usa los tres.
        k: Número de top variables en cada método.

    Returns:
        DataFrame con columnas ['variable', 'method', 'score'], ordenado desc.
    """
    if methods is None:
        methods = ['corr', 'mi', 'lasso']
    records = []
    for m in methods:
        if m == 'corr':
            corrs = X.apply(lambda col: np.abs(np.corrcoef(col.fillna(col.mean()), y.fillna(y.mean()))[0, 1]))
            top = corrs.sort_values(ascending=False).head(k)
            for var, score in top.items():
                records.append((var, 'corr', score))
        elif m == 'mi':
            mi = mutual_info_regression(X.fillna(X.mean()), y.fillna(y.mean()))
            mi_series = pd.Series(mi, index=X.columns)
            top = mi_series.sort_values(ascending=False).head(k)
            for var, score in top.items():
                records.append((var, 'mi', score))
        elif m == 'lasso':
            lasso = LassoCV(cv=5, max_iter=10000).fit(X.fillna(X.mean()), y.fillna(y.mean()))
            coef = pd.Series(np.abs(lasso.coef_), index=X.columns)
            top = coef.sort_values(ascending=False).head(k)
            for var, score in top.items():
                records.append((var, 'lasso', score))
        else:
            raise ValueError(f"Método desconocido: {m}")
    df_rank = pd.DataFrame(records, columns=['variable', 'method', 'score'])
    return df_rank.sort_values('score', ascending=False).reset_index(drop=True)
