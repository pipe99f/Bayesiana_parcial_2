import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from multinivel_bayesiano.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
)

# Leer datos
dtype_dict = {"COD_MCPIO": str, "COD_DEPTO": str}
saber_11 = pd.read_csv(PROCESSED_DATA_DIR / "saber11.csv", dtype=dtype_dict)
cns = pd.read_csv(PROCESSED_DATA_DIR / "cns.csv", dtype=dtype_dict)

# %% Mapas departamentos
colombia_depto = gpd.read_file(EXTERNAL_DATA_DIR / "colombia_depto.shp")
colombia_depto = colombia_depto.query("dpto_ccdgo not in ['88']")

media_depto = saber_11.groupby("COD_DEPTO")["PUNT_GLOBAL"].mean().reset_index()
colombia_depto = colombia_depto.merge(
    media_depto, left_on="dpto_ccdgo", right_on="COD_DEPTO", how="left"
)
colombia_depto.plot(
    column="PUNT_GLOBAL",
    cmap="YlGnBu",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Media"},
    missing_kwds={"color": "lightgrey"},
)
plt.show()


# %% Mapas municipios - Media muestral y cobertura neta año 2022
colombia_mcpio = gpd.read_file(EXTERNAL_DATA_DIR / "colombia_mcpio.shp")
colombia_mcpio = colombia_mcpio.query(
    # Eliminar San Andrés y Providencia
    "mpio_ccnct not in ['88564', '88001']"
)


## Media muestral puntaje global
media_mcpio = saber_11.groupby("COD_MCPIO")["PUNT_GLOBAL"].mean().reset_index()
colombia_mcpio = colombia_mcpio.merge(
    media_mcpio, left_on="mpio_ccnct", right_on="COD_MCPIO", how="left"
)
colombia_mcpio.plot(
    column="PUNT_GLOBAL",
    cmap="YlGnBu",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Media"},
    missing_kwds={"color": "lightgrey"},
)
plt.show()

## Cobertura neta secundaria 2022 por municipio
colombia_mcpio = colombia_mcpio.merge(
    cns, left_on="mpio_ccnct", right_on="COD_MCPIO", how="left"
)
colombia_mcpio.plot(
    column="CNS",
    cmap="YlGnBu",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Media"},
    missing_kwds={"color": "lightgrey"},
)
plt.show()
