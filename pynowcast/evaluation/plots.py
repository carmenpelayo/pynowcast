import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(y_true: pd.Series, y_pred: pd.Series, title: str = None) -> None:
    """
    Grafica series de valores reales vs pronósticos.

    Args:
        y_true: Serie de valores reales indexada por fecha.
        y_pred: Serie de valores pronosticados indexada por fecha.
        title: Título del gráfico.
    """
    plt.figure()
    plt.plot(y_true.index, y_true.values, label='Real')
    plt.plot(y_pred.index, y_pred.values, label='Pronóstico')
    plt.legend()
    if title:
        plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.tight_layout()
    plt.show()


def plot_metrics_bar(df_metrics: pd.DataFrame, metric: str, title: str = None) -> None:
    """
    Grafica un diagrama de barras para una métrica de evaluación dada.

    Args:
        df_metrics: DataFrame con métricas indexadas por serie.
        metric: Nombre de la columna de métrica a graficar.
        title: Título del gráfico.
    """
    plt.figure()
    df_metrics[metric].plot(kind='bar')
    if title:
        plt.title(title)
    plt.xlabel('Serie')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()


def plot_error_heatmap(df_errors: pd.DataFrame, title: str = None) -> None:
    """
    Muestra un heatmap de errores (o métricas) para múltiples series y horizontes.

    Args:
        df_errors: DataFrame donde filas y columnas representan dimensiones de error.
        title: Título del gráfico.
    """
    plt.figure()
    im = plt.imshow(df_errors.values, aspect='auto')
    plt.colorbar(im)
    plt.xticks(range(len(df_errors.columns)), df_errors.columns, rotation=45)
    plt.yticks(range(len(df_errors.index)), df_errors.index)
    if title:
        plt.title(title)
    plt.xlabel('Horizonte / Serie')
    plt.ylabel('Serie / Horizonte')
    plt.tight_layout()
    plt.show()
