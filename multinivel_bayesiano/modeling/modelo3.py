import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import gammaln
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

cod_mcpio = saber_11[
    "COD_MCPIO"
].to_numpy()  # Se convierten los códigos de los mcpios a NumPy


stats_departamento = saber_11.groupby("COD_DEPTO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)

nj = stats_departamento["count"].values  # Número de muestras por departamento
ybj = stats_departamento["mean"].values  # Media de puntaje global por departamento
s2j = stats_departamento["var"].values  # Varianza de puntaje global por departamento


stats_municipio = saber_11.groupby("COD_MCPIO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)
njk = stats_municipio["count"].values
ybk = stats_municipio["mean"].values
s2k = stats_municipio["var"].values

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


# %% MCMC3
## Para este punto cambia la notación de ybj y ybk,
## j indica municipios y k indica departamentos
stats_departamento = saber_11.groupby("COD_DEPTO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)

nk = stats_departamento["count"].values  # Número de muestras por departamento
ybk = stats_departamento["mean"].values  # Media de puntaje global por departamento
s2k = stats_departamento["var"].values  # Varianza de puntaje global por departamento

indice_depto, cod_depto = pd.factorize(saber_11["COD_DEPTO"])

stats_municipio = saber_11.groupby("COD_MCPIO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)
nj = stats_municipio["count"].values
ybj = stats_municipio["mean"].values
s2j = stats_municipio["var"].values

indice_mcpio, cod_mcpio = pd.factorize(saber_11["COD_MCPIO"])

mun_depto = np.array([len(np.unique(indice_mcpio[indice_depto == idx])) for idx in range(len(cod_depto))])
indice_mun_depto = np.repeat(np.unique(indice_depto), mun_depto)


def MCMC3(B, njk, y, cod_depto, cod_mcpio, s2j):
    # Hiperparámetros
    upsilon = 3
    mu_0 = 250
    gamma2_0 = 50**2
    eta_0 = 1
    tau2_0 = 50**2
    nu_0 = 1
    sigma2_0 = 50**2
    xi_0 = 1
    kappa2_0 = 50**2

    # Valores iniciales
    n = len(y)
    m = len(cod_depto)
    nk = len(cod_mcpio)
    sigma2 = 1 / np.random.gamma(0.5, 2 / 50**2)
    tau2 = 1 / np.random.gamma(0.5, 2 / 50**2)
    mu = np.random.normal(250, 50)
    theta = np.random.normal(mu, np.sqrt(tau2), m)
    kappa2 = 1 / np.random.gamma(0.5, 2 / 50**2)
    zeta = np.random.normal(theta[indice_mun_depto], np.sqrt(sigma2), nk)
    varsigma2 = 1 / np.random.gamma((upsilon + 1)*0.5,
                                    2 / ((y - zeta[indice_mcpio])**2 + upsilon * kappa2))
    sum_inv_varsigma2_jk = np.zeros(nk)  # para zeta_jk
    sum_yijk_inv_varsigma2_jk = np.zeros(nk)  # para zeta_jk
    sum_zeta_div_sigma2 = np.zeros(m)  # para theta_k

    # Almacenamiento
    muestras = {}
    muestras["zeta"] = []
    muestras["theta"] = []
    muestras["kappa2"] = []
    muestras["varsigma2"] = []
    muestras["sigma2"] = []
    muestras["mu"] = []
    muestras["tau2"] = []

    LL = []

    for b in tqdm(range(B), desc="MCMC3 Procesando"):
        # Muestreo de zeta_{j,k}
        np.add.at(sum_inv_varsigma2_jk, indice_mcpio, 1.0 / varsigma2)
        np.add.at(sum_yijk_inv_varsigma2_jk, indice_mcpio, y / varsigma2)
        var_zeta = 1 / (sum_inv_varsigma2_jk + 1 / sigma2)
        mean_zeta = var_zeta * (sum_yijk_inv_varsigma2_jk + theta[indice_mun_depto] / sigma2)
        zeta = np.random.normal(mean_zeta, np.sqrt(var_zeta))

        # Muestreo de theta_k
        np.add.at(sum_zeta_div_sigma2, indice_mun_depto, zeta)
        var_theta = 1 / ( mun_depto / sigma2 + 1 / tau2)
        mean_theta = var_theta * (sum_zeta_div_sigma2 / sigma2 + mu / tau2)
        theta = np.random.normal(mean_theta, np.sqrt(var_theta))

        # Muestreo de mu
        var_mu = 1 / (m / tau2 + 1 / gamma2_0)
        mean_mu = var_mu * (np.sum(theta) / tau2 + mu_0 / gamma2_0)
        mu = np.random.normal(mean_mu, np.sqrt(var_mu))

        # Muestreo de tau^2
        tau2_n = 33 * tau2_0 + np.sum((theta - mu) ** 2)
        tau2 = 1 / np.random.gamma(33 * 0.5, 2 / tau2_n)

        # Muestreo de sigma^2
        sigma2_n = 1113 * sigma2_0 + np.sum(
            (zeta - theta[indice_mun_depto]) ** 2
        )
        sigma2 = 1 / np.random.gamma(1113 * 0.5, 2 / sigma2_n)

        # Muestreo de varsigma^2_{i,j,k}
        kappa2_n = upsilon * kappa2 + (y - zeta[indice_mcpio])**2
        varsigma2 = 1 / np.random.gamma((upsilon + 1) * 0.5, 2 / kappa2_n)

        # Muestreo de kappa^2
        xi_n = xi_0 + upsilon * n
        kappa2_n = xi_0 * kappa2_0 + upsilon * np.sum(1 / varsigma2)
        kappa2 = 1 / np.random.gamma(xi_n * 0.5, 2 / kappa2_n)

        # Guardar muestras
        if (b % 10) == 0:
            muestras["zeta"].append(zeta)
            muestras["theta"].append(theta)
            muestras["kappa2"].append(kappa2)
            muestras["varsigma2"].append(varsigma2)
            muestras["sigma2"].append(sigma2)
            muestras["mu"].append(mu)
            muestras["tau2"].append(tau2)

            # Log-verosimilitud (adaptado para el nuevo modelo)
            LL.append(
                np.sum(st.norm.logpdf(x=y, loc=zeta[indice_mcpio], scale=np.sqrt(varsigma2)))
            )

    return {"muestras": muestras, "LL": LL}




# %% Inicializar el muestreador MCMC3
np.random.seed(123)
muestras_mcmc3 = MCMC3(
    B=10000, y=y, cod_depto=cod_depto, s2j=s2j, njk=njk, cod_mcpio=cod_mcpio
)
