import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import typer

from multinivel_bayesiano.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    pobreza_monetaria_input_path: Path = RAW_DATA_DIR / "pobreza monetaria.xls",
    cns_input_path: Path = RAW_DATA_DIR / "estadísticas educación.csv",
    pobreza_montetaria_output_path: Path = PROCESSED_DATA_DIR / "pobreza_monetaria.csv",
    cns_output_path: Path = PROCESSED_DATA_DIR / "cns.csv",
    shapefile_mcpio_path: Path = EXTERNAL_DATA_DIR / "colombia_mcpio.shp",
    shapefile_depto_path: Path = EXTERNAL_DATA_DIR / "colombia_depto.shp",
    # ----------------------------------------------
):
    pobreza_monetaria_2018 = pd.read_excel(
        pobreza_monetaria_input_path,
        sheet_name="Pobreza Monetaria (%)",
        header=None,
    )

    # Columna 15 (año 2018)
    pobreza_data = pobreza_monetaria_2018.iloc[1:, 15].values.tolist()
    # Columna 0 (nombre municipios)
    pobreza_DPTO = pobreza_monetaria_2018.iloc[1:, 0].values.tolist()

    pobreza_monetaria = pd.DataFrame({"INCIDENCIA": pobreza_data, "DPTO": pobreza_DPTO})
    cod_depto_map = {
        "Antioquia": "05",
        "Atlántico": "08",
        "Bogotá D.C.": "11",
        "Bolívar": "13",
        "Boyacá": "15",
        "Caldas": "17",
        "Caquetá": "18",
        "Cauca": "19",
        "Cesar": "20",
        "Chocó": "27",
        "Córdoba": "23",
        "Cundinamarca": "25",
        "Huila": "41",
        "La Guajira": "44",
        "Magdalena": "47",
        "Meta": "50",
        "Nariño": "52",
        "Norte de Santander": "54",
        "Quindío": "63",
        "Risaralda": "66",
        "Santander": "68",
        "Sucre": "70",
        "Tolima": "73",
        "Valle del Cauca": "76",
    }
    pobreza_monetaria["COD_DEPTO"] = pobreza_monetaria["DPTO"].map(cod_depto_map)

    cns = pd.read_csv(cns_input_path)
    cns = cns[["COBERTURA_NETA_SECUNDARIA", "CÓDIGO_MUNICIPIO", "MUNICIPIO"]]
    cns.columns = ["CNS", "COD_MCPIO", "MCPIO"]
    cns = cns.dropna()
    cns["COD_MCPIO"] = cns["COD_MCPIO"].astype(str).str.zfill(5)

    pobreza_monetaria.to_csv(pobreza_montetaria_output_path, index=False)
    cns.to_csv(cns_output_path, index=False)

    # Descarga shapefile departamentos
    url_shp_depto = "https://geoserver.dane.gov.co/geoserver/geoportal/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=geoportal:mgn2022_dpto&outputFormat=application/json"
    if not os.path.exists(shapefile_depto_path):
        try:
            colombia = gpd.read_file(url_shp_depto)
            colombia.to_file(shapefile_depto_path)
            print(f"Shapefile descargada y guardada {shapefile_depto_path}")
        except Exception as e:
            print(f"Error descargando la shapefile: {e}")
    else:
        print(f"Shapefile ya existe en {shapefile_depto_path}")

    # Descarga shapefile municipios
    url_shp_mcpio = "https://geoserver.dane.gov.co/geoserver/geoportal/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=geoportal:mgn2018_mpio&outputFormat=application/json"

    if not os.path.exists(shapefile_mcpio_path):
        try:
            colombia = gpd.read_file(url_shp_mcpio)
            colombia.to_file(shapefile_mcpio_path)
            print(f"Shapefile descargada y guardada {shapefile_mcpio_path}")
        except Exception as e:
            print(f"Error descargando la shapefile: {e}")
    else:
        print(f"Shapefile ya existe en {shapefile_mcpio_path}")


if __name__ == "__main__":
    app()
