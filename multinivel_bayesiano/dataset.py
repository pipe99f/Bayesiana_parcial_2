from pathlib import Path

import pandas as pd
import typer

from multinivel_bayesiano.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    pobreza_monetaria_input_path: Path = RAW_DATA_DIR / "pobreza monetaria.xls",
    cns_input_path: Path = RAW_DATA_DIR / "estadísticas educación.csv",
    pobreza_montetaria_output_path: Path = PROCESSED_DATA_DIR / "pobreza_monetaria.csv",
    cns_output_path: Path = PROCESSED_DATA_DIR / "cns.csv",
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

    cns = pd.read_csv(cns_input_path)
    cns = cns[["COBERTURA_NETA_SECUNDARIA", "CÓDIGO_MUNICIPIO", "MUNICIPIO"]]
    cns.columns = ["CNS", "COD_MCPIO", "MCPIO"]
    cns = cns.dropna()

    pobreza_monetaria.to_csv(pobreza_montetaria_output_path, index=False)
    cns.to_csv(cns_output_path, index=False)


if __name__ == "__main__":
    app()
