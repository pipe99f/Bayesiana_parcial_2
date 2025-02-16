import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

from multinivel_bayesiano.config import PROCESSED_DATA_DIR

# Leer datos
dtype_dict = {"COD_MCPIO": str, "COD_DEPTO": str}
saber_11 = pd.read_csv(PROCESSED_DATA_DIR / "saber11.csv", dtype=dtype_dict)

# %% Preparar datos
saber_11 = saber_11.sort_values(by=["COD_DEPTO", "COD_MCPIO"], ascending=[True, True])
saber_11

y = saber_11["PUNT_GLOBAL"].values  # Se convierte a NumPy para optimizar
n = len(saber_11)  # Número de estudiantes: 525061
m = len(saber_11["COD_DEPTO"].unique())  # Número de departamentos: 32
nk = len(saber_11["COD_MCPIO"].unique())  # Número de municipios: 1112
Y = [[] for _ in range(m)]  # Lista de listas vacías de tamaño m
g = np.full(n, np.nan)  # Vector de tamaño n con valores NaN
cod_depto = saber_11[
    "COD_DEPTO"
].to_numpy()  # Se convierten los códigos de departamento a NumPy


stats_departamento = saber_11.groupby("COD_DEPTO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)

nj = stats_departamento["count"].values
ybj = stats_departamento["mean"].values
s2j = stats_departamento["var"].values


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

# %% MCMC1


# @njit
def MCMC1(B, nj, y, cod_depto, s2j):
    # Hiperparámetros
    upsilon = 3
    mu_0 = 250
    gamma2_0 = 50**2
    eta_0 = 1
    tau2_0 = 50**2
    nu_0 = 1
    sigma2_0 = 50**2

    # valores iniciales
    n = len(y)
    theta = ybj
    varsigma2 = np.ones(n)
    sigma2 = np.mean(s2j)
    mu = np.mean(theta)
    tau2 = np.var(theta, ddof=1)

    # indexar varsigma2 por departamento
    unique_deptos = np.unique(cod_depto)
    depto_to_index = {depto: i for i, depto in enumerate(unique_deptos)}
    depto_indices = np.vectorize(depto_to_index.get)(cod_depto)

    sum_inv_varsigma2 = np.zeros(32)
    sum_yij_inv_varsigma2 = np.zeros(32)

    # Almacenamiento
    muestras = {}
    muestras["theta"] = []
    muestras["sigma2"] = []
    muestras["mu"] = []
    muestras["tau2"] = []

    LL = []

    for b in tqdm(range(B), desc="Procesando"):
        # for b in range(B):
        # Muestreo de theta_{j}
        # groupby depto and sum 1/varsigma2
        np.add.at(sum_inv_varsigma2, depto_indices, 1.0 / varsigma2)

        # groupby depto and sum y_ij/varsigma2
        np.add.at(sum_yij_inv_varsigma2, depto_indices, y / varsigma2)

        var_theta = 1 / (sum_inv_varsigma2 + 1 / tau2)
        mean_theta = var_theta * (sum_yij_inv_varsigma2 + mu / tau2)
        theta = np.random.normal(mean_theta, np.sqrt(var_theta))

        # Muestreo de sigma^2
        nu_n = nu_0 + n * upsilon
        sigma2_n = nu_0 * sigma2_0 + upsilon * np.sum(sum_inv_varsigma2)
        sigma2 = 1 / np.random.gamma(
            nu_n / 2,
            sigma2_n / 2,
        )

        # Muestreo de varsigma^2_{i,j}
        extended_theta = np.repeat(theta, nj)
        varsigma2 = 1 / np.random.gamma(
            (upsilon + 1) / 2, ((y - extended_theta) ** 2 + upsilon * sigma2) / 2
        )

        # Muestreo de mu
        var_mu = 1 / (m / tau2 + 1 / gamma2_0)
        mean_mu = var_mu * (np.sum(theta) / tau2 + mu_0 / gamma2_0)
        mu = np.random.normal(mean_mu, np.sqrt(var_mu))

        # Muestreo de tau^2
        eta_n = eta_0 + m
        tau2_n = (eta_0 * tau2_0 + np.sum((theta - mu) ** 2)) / 2
        tau2 = 1 / np.random.gamma(eta_n / 2, tau2_n)

        # Guardar muestras
        if (b % 10) == 0:
            muestras["theta"].append(theta)
            muestras["sigma2"].append(sigma2)
            muestras["mu"].append(mu)
            muestras["tau2"].append(tau2)

            # log-verosimilitud
            LL.append(
                np.sum(
                    st.norm.logpdf(x=y, loc=np.repeat(theta, nj), scale=np.sqrt(sigma2))
                )
            )
            # print(b)
    return {"muestras": muestras, "LL": LL}


muestras_mcmc1 = MCMC1(B=10000, nj=nj, y=y, cod_depto=cod_depto, s2j=s2j)


muestras_mcmc1["muestras"]["theta"]
muestras_mcmc1["muestras"]["sigma2"]
muestras_mcmc1["muestras"]["mu"]
muestras_mcmc1["muestras"]["tau2"]


# %% MCMC2

# %% MCMC3

stats_municipio = saber_11.groupby("COD_MCPIO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)
njk = stats_municipio["count"].values
ybk = stats_municipio["mean"].values
s2k = stats_municipio["var"].values

# %% MCMC4


if __name__ == "__main__":
    MCMC1()
