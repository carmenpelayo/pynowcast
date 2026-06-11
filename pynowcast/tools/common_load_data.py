import pandas as pd
import numpy as np


def common_load_data(
    excel_datafile: str,
    mon_freq: str,
    quar_freq: str,
    blocks_sheet: str,
    Par,
    m: int,
    do_loop: bool,
    date_today,
    Loop
):
    """
    Carga datos de Excel y prepara inputs para estimación.

    Args:
        excel_datafile: Ruta al archivo de datos (sin extensión).
        mon_freq: Nombre de la hoja de datos mensuales.
        quar_freq: Nombre de la hoja de datos trimestrales.
        blocks_sheet: Nombre de la hoja de bloques y grupos.
        Par: namespace de parámetros del modelo (se actualiza con nM, nQ, blocks).
        m: número de meses ahead (p.ej. 6).
        do_loop: flag para bucle de modelos.
        date_today: fecha de corte para evaluación o nowcast.
        Loop: namespace de parámetros del bucle.

    Returns:
        Par: con campos nM, nQ, blocks.
        xest: DataFrame con series concatenadas (mensuales + trimestrales).
        t_m: entero, número de meses del trimestre para GDP availability.
        groups: ndarray con identificación de grupo para cada serie.
        nameseries: lista de nombres mnemotécnicos de series.
        blocks: ndarray con estructura de bloques.
        groups_name: lista de nombres de grupos.
        fullnames: lista de nombres completos de series.
        datet: ndarray T x 2 con [year, month] para cada observación.
        Loop: actualizado (nombre de loop si do_loop==2).
    """
    # Construir ruta con extensión .xlsx si no la tiene
    file_xlsx = excel_datafile if excel_datafile.endswith('.xlsx') else excel_datafile + '.xlsx'

    # Leer datos mensuales y trimestrales
    df_mon = pd.read_excel(file_xlsx, sheet_name=mon_freq, parse_dates=[0], index_col=0)
    df_quar = pd.read_excel(file_xlsx, sheet_name=quar_freq, parse_dates=[0], index_col=0)

    # Leer configuración de bloques/grupos
    df_blocks = pd.read_excel(file_xlsx, sheet_name=blocks_sheet)

    # Concatenar series: primero mensuales luego trimestrales
    xest = pd.concat([df_mon, df_quar], axis=1)

    # Actualizar parámetros de Par
    Par.nM = df_mon.shape[1]
    Par.nQ = df_quar.shape[1]
    # Bloques: matriz bloques (una fila por serie)
    # Suponemos df_blocks tiene columna 'block'
    blocks = df_blocks['block'].values if 'block' in df_blocks else np.zeros(xest.shape[1], dtype=int)
    Par.blocks = blocks

    # Grupos y nombres de series
    groups = blocks
    nameseries = xest.columns.tolist()
    fullnames = df_blocks['full_name'].tolist() if 'full_name' in df_blocks else nameseries
    groups_name = df_blocks['group_name'].unique().tolist() if 'group_name' in df_blocks else list(np.unique(groups))

    # Fechas en formato [year, month]
    datet = np.vstack([xest.index.year, xest.index.month]).T

    # t_m: mes del trimestre de GDP availability
    t_m = m  # por defecto m meses ahead

    # Actualizar Loop.name_loop si do_loop == 2 ya se hace en main

    return Par, xest, t_m, groups, nameseries, blocks, groups_name, fullnames, datet, Loop
