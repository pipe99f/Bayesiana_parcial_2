# from loguru import logger
import numpy as np
import pandas as pd

from multinivel_bayesiano.config import PROCESSED_DATA_DIR

# Leer datos
dtype_dict = {"COD_MCPIO": str, "COD_DEPTO": str}
saber_11 = pd.read_csv(PROCESSED_DATA_DIR / "saber11.csv", dtype=dtype_dict)

# %% Preparar datos
saber_11 = saber_11.sort_values(by=["COD_DEPTO", "COD_MCPIO"], ascending=[True, True])
saber_11

y = saber_11["PUNT_GLOBAL"].values  # Se convierte a NumPy para optimizar
m = len(saber_11["COD_DEPTO"].unique())  # Número de departamentos: 32
n = len(saber_11)  # Número de estudiantes: 525061
Y = [[] for _ in range(m)]  # Lista de listas vacías de tamaño m
g = np.full(n, np.nan)  # Vector de tamaño n con valores NaN
deptos = sorted(saber_11["COD_DEPTO"].unique())  # Departamentos ordenados

for j in range(m):
    dept_code = deptos[j]
    idx = saber_11["ESTU_COD_RESIDE_DEPTO"] == dept_code
    g[idx] = j
    Y[j] = y[idx].tolist()  # Convert the pandas series to a python list


# %% Hiperparámetros
upsilon = 3
mu_0 = 250
gamma2_0 = 50**2
eta_0 = 1
tau2_0 = 50**2
nu_0 = 1
sigma2_0 = 50**2
lambda_0 = 1
alpha_0 = 1
beta_0 = 1 / 50**2
xi_0 = 1
kappa2_0 = 50**2
len(saber_11)

# %%

if __name__ == "__main__":
    MCMC1()
