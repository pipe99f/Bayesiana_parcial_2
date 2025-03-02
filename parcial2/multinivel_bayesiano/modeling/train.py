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

# %% MCMC1


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
    sigma2 = np.mean(s2j)
    varsigma2 = np.repeat(sigma2, n)
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
        # Muestreo de theta_{j}
        # groupby depto and sum 1/varsigma2

        sum_inv_varsigma2 = np.zeros(32)
        np.add.at(sum_inv_varsigma2, depto_indices, 1.0 / varsigma2)

        # groupby depto and sum y_ij/varsigma2
        sum_yij_inv_varsigma2 = np.zeros(32)
        np.add.at(sum_yij_inv_varsigma2, depto_indices, y / varsigma2)

        var_theta = 1 / (sum_inv_varsigma2 + 1 / tau2)
        mean_theta = var_theta * (sum_yij_inv_varsigma2 + mu / tau2)
        theta = np.random.normal(mean_theta, np.sqrt(var_theta))

        # Muestreo de sigma^2
        nu_n = nu_0 + n * upsilon
        sigma2_n = nu_0 * sigma2_0 + upsilon * np.sum(sum_inv_varsigma2)
        sigma2 = 1 / np.random.gamma(
            nu_n / 2,
            2 / sigma2_n,  # scale
        )

        # Muestreo de varsigma^2_{i,j}
        extended_theta = np.repeat(theta, nj)
        varsigma2 = 1 / np.random.gamma(
            (upsilon + 1) / 2,
            2 / ((y - extended_theta) ** 2 + upsilon * sigma2),  # scale
        )

        # Muestreo de mu
        var_mu = 1 / (m / tau2 + 1 / gamma2_0)
        mean_mu = var_mu * (np.sum(theta) / tau2 + mu_0 / gamma2_0)
        mu = np.random.normal(mean_mu, np.sqrt(var_mu))

        # Muestreo de tau^2
        eta_n = eta_0 + m
        tau2_n = eta_0 * tau2_0 + np.sum((theta - mu) ** 2)
        tau2 = 1 / np.random.gamma(
            eta_n / 2,
            2 / tau2_n,  # scale
        )

        # Guardar muestras
        if (b % 10) == 0:
            muestras["theta"].append(theta)
            muestras["sigma2"].append(sigma2)
            muestras["mu"].append(mu)
            muestras["tau2"].append(tau2)

            # log-verosimilitud
            LL.append(
                np.sum(
                    st.t.logpdf(x=y, df=nu_0, loc=np.repeat(theta, nj), scale=sigma2)
                )
            )
    return {"muestras": muestras, "LL": LL}


# %% Inicializar el muestreador MCMC1
np.random.seed(123)
muestras_mcmc1 = MCMC1(B=110000, nj=nj, y=y, cod_depto=cod_depto, s2j=s2j)

# %% Data frame de theta
mcmc1_theta = pd.DataFrame(muestras_mcmc1["muestras"]["theta"])
mcmc1_theta = mcmc1_theta[1000:]
mcmc1_theta.to_csv(PROCESSED_DATA_DIR / "mcmc1_theta.csv")

# Data frame con el resto de parámetros
mcmc1_df = pd.DataFrame(muestras_mcmc1["muestras"])
mcmc1_df = mcmc1_df.drop(columns=["theta"])
mcmc1_df = mcmc1_df[1000:]
mcmc1_df.to_csv(PROCESSED_DATA_DIR / "mcmc1_df.csv")

# log-verosimilitud
mcmc1_ll = pd.DataFrame(muestras_mcmc1["LL"])
mcmc1_ll = mcmc1_ll[1000:]
mcmc1_ll.to_csv(PROCESSED_DATA_DIR / "mcmc1_ll.csv")


# %% MCMC2
def MCMC2(
    B,
    y,
    cod_depto,
    nj,
    ybj,
    s2j,
    upsilon,
    mu_0,
    gamma2_0,
    eta_0,
    tau2_0,
    lambda_0,
    alpha_0,
    beta_0,
    sigma2_0,
):
    n = np.sum(nj)
    m = len(nj)
    nu_0 = np.arange(1, 51, dtype=int)
    sigma2_j = s2j
    theta = ybj
    mu = mu_0
    sigma2 = sigma2_0
    tau2 = tau2_0
    nu = 1
    varsigma2 = np.ones(n)
    # indexar varsigma2 por departamento
    unique_deptos = np.unique(cod_depto)
    depto_to_index = {depto: i for i, depto in enumerate(unique_deptos)}
    depto_indices = np.vectorize(depto_to_index.get)(cod_depto)

    sum_inv_varsigma2 = np.zeros(32)
    sum_yij_inv_varsigma2 = np.zeros(32)
    THETA = np.zeros((B, 2 * m + 5))
    for j in tqdm(range(B), desc="Procesando"):
        np.add.at(sum_inv_varsigma2, depto_indices, 1.0 / varsigma2)

        # groupby depto and sum y_ij/varsigma2
        np.add.at(sum_yij_inv_varsigma2, depto_indices, y / varsigma2)

        vtheta = 1 / (sum_inv_varsigma2 + 1 / tau2)
        # Se actualiza theta
        theta = np.random.normal(
            (sum_yij_inv_varsigma2 + mu / tau2) * vtheta, np.sqrt(vtheta), m
        )
        # Se actualiza varsigma
        varsigma2_scale = 2 / (
            (y - np.repeat(theta, nj)) ** 2 + upsilon * np.repeat(sigma2_j, nj)
        )
        varsigma2 = 1 / np.random.gamma((upsilon + 1) * 0.5, varsigma2_scale)
        # Se actualiza sigma2_j
        vsigma2_j = upsilon * sum_inv_varsigma2
        sigma2_j_scale = 2 / (nu * sigma2 + vsigma2_j)
        sigma2_j = 1 / np.random.gamma((nu + upsilon * nj) * 0.5, sigma2_j_scale, m)
        # Se actualiza tau^2
        tau2_scale = 2 / (eta_0 * tau2_0 + m * (np.mean(theta) - mu) ** 2)
        tau2 = 1 / np.random.gamma((eta_0 + m) * 0.5, tau2_scale)
        # Se actualiza mu
        vmu = 1 / (m / tau2 + 1 / gamma2_0)
        mu = np.random.normal(
            (m * np.mean(theta) / tau2 + mu_0 / gamma2_0) * vmu, np.sqrt(vmu)
        )
        # Se actualiza sigma_2
        sigma2_scale = 2 / (beta_0 + nu * np.sum(1 / sigma2_j))
        sigma2 = 1 / np.random.gamma((alpha_0 + nu * m) * 0.5, sigma2_scale)
        # Se actualiza nu
        vnu = (
            0.5 * m * nu_0 * np.log(0.5 * nu_0 * sigma2)
            - m * gammaln(0.5 * nu_0)
            - 0.5 * nu_0 * np.sum(np.log(sigma2_j))
            - nu_0 * (lambda_0 + 0.5 * sigma2 * np.sum(1 / sigma2_j))
        )
        prob_nu = np.exp(vnu - np.max(vnu)) / np.sum(np.exp(vnu - np.max(vnu)))
        nu = np.random.choice(nu_0, p=prob_nu)
        # Log-verosimilitud
        logver = np.sum(
            st.norm.logpdf(y, np.repeat(theta, nj), np.sqrt(np.repeat(sigma2_j, nj)))
        )
        THETA[j] = np.r_[theta, sigma2_j, sigma2, mu, tau2, nu, logver]

    return THETA


muestras_mcmc2 = MCMC2(
    1000,
    y,
    cod_depto,
    nj,
    ybj,
    s2j,
    upsilon,
    mu_0,
    gamma2_0,
    eta_0,
    tau2_0,
    lambda_0,
    alpha_0,
    beta_0,
    sigma2_0,
)


def MCMC2(B, nj, y, cod_depto, s2j):
    # Hiperparámetros
    upsilon = 3
    mu_0 = 250
    gamma2_0 = 50**2
    eta_0 = 1
    tau2_0 = 50**2
    lambda_0 = 1

    # valores iniciales
    nu = 1
    n = len(y)
    theta = ybj
    varsigma2 = np.ones(n)
    sigma2 = np.mean(s2j)
    mu = np.mean(theta)
    tau2 = np.var(theta, ddof=1)
    nu_0 = np.arange(1, 51, dtype=int)

    # indexar varsigma2 por departamento
    unique_deptos = np.unique(cod_depto)
    depto_to_index = {depto: i for i, depto in enumerate(unique_deptos)}
    depto_indices = np.vectorize(depto_to_index.get)(cod_depto)

    sum_inv_varsigma2 = np.zeros(32)
    sum_yij_inv_varsigma2 = np.zeros(32)

    # Almacenamiento
    muestras = {}
    muestras["theta"] = []
    muestras["sigma2_j"] = []
    muestras["sigma2"] = []
    muestras["mu"] = []
    muestras["nu"] = []
    muestras["tau2"] = []

    LL = []

    for b in tqdm(range(B), desc="Procesando"):
        # Muestreo de theta_{j}
        # groupby depto and sum 1/varsigma2
        sum_inv_varsigma2 = np.zeros(32)
        np.add.at(sum_inv_varsigma2, depto_indices, 1.0 / varsigma2)

        # groupby depto and sum y_ij/varsigma2

        sum_yij_inv_varsigma2 = np.zeros(32)
        np.add.at(sum_yij_inv_varsigma2, depto_indices, y / varsigma2)

        var_theta = 1 / (sum_inv_varsigma2 + 1 / tau2)
        mean_theta = var_theta * (sum_yij_inv_varsigma2 + mu / tau2)
        theta = np.random.normal(mean_theta, np.sqrt(var_theta))

        # Muestreo de sigma2_j
        nu_n = nu + nj * upsilon
        sigma2_n = nu * sigma2 + upsilon * sum_inv_varsigma2
        sigma2_j = 1 / np.random.gamma(
            nu_n / 2,
            2 / sigma2_n,  # scale
        )

        # Muestreo de sigma^2
        alpha_n = alpha_0 + nu * m
        beta_n = beta_0 + nu * np.sum(1 / sigma2_j)
        sigma2 = 1 / np.random.gamma(
            alpha_n / 2,
            2 / beta_n,  # scale
        )

        # Muestreo de nu
        lpnu = (
            0.5 * m * nu_0 * np.log(0.5 * nu_0 * sigma2)
            - m * gammaln(0.5 * nu_0)
            - 0.5 * nu_0 * np.sum(np.log(sigma2_j))
            - nu_0 * (lambda_0 + 0.5 * sigma2 * np.sum(1 / sigma2_j))
        )
        prob_nu = np.exp(lpnu - np.max(lpnu)) / np.sum(np.exp(lpnu - np.max(lpnu)))
        # prob_nu = np.exp(lpnu - np.max(lpnu))
        nu = np.random.choice(nu_0, p=prob_nu)

        # Muestreo de varsigma^2_{i,j}
        extended_theta = np.repeat(theta, nj)
        varsigma2 = 1 / np.random.gamma(
            (upsilon + 1) / 2,
            2 / ((y - extended_theta) ** 2 + upsilon * sigma2),  # scale
        )

        # Muestreo de mu
        var_mu = 1 / (m / tau2 + 1 / gamma2_0)
        mean_mu = var_mu * (np.sum(theta) / tau2 + mu_0 / gamma2_0)
        mu = np.random.normal(mean_mu, np.sqrt(var_mu))

        # Muestreo de tau^2
        eta_n = eta_0 + m
        tau2_n = eta_0 * tau2_0 + np.sum((theta - mu) ** 2)
        tau2 = 1 / np.random.gamma(
            eta_n / 2,
            2 / tau2_n,  # scale
        )

        # Guardar muestras
        if (b % 10) == 0:
            muestras["theta"].append(theta)
            muestras["sigma2_j"].append(sigma2_j)
            muestras["sigma2"].append(sigma2)
            muestras["nu"].append(nu)
            muestras["mu"].append(mu)
            muestras["tau2"].append(tau2)

            # log-verosimilitud
            LL.append(
                np.sum(
                    st.norm.logpdf(
                        x=y, loc=np.repeat(theta, nj), scale=np.sqrt(varsigma2)
                    )
                )
            )
            # print(b)
    return {"muestras": muestras, "LL": LL}


# %% Inicializar el muestreador MCMC2
np.random.seed(123)
muestras_mcmc2 = MCMC2(B=110000, nj=nj, y=y, cod_depto=cod_depto, s2j=s2j)


# %% Data frame de theta
mcmc2_theta = pd.DataFrame(muestras_mcmc2["muestras"]["theta"])
mcmc2_theta = mcmc2_theta[1000:]
mcmc2_theta.to_csv(PROCESSED_DATA_DIR / "mcmc2_theta.csv")

# %% Data frame de sigma2_j
mcmc2_sigma2_j = pd.DataFrame(muestras_mcmc2["muestras"]["sigma2_j"])
mcmc2_sigma2_j
mcmc2_sigma2_j = mcmc2_sigma2_j[1000:]
mcmc2_sigma2_j.to_csv(PROCESSED_DATA_DIR / "mcmc2_sigma2_j.csv")

# Data frame con el resto de parámetros
mcmc2_df = pd.DataFrame(muestras_mcmc2["muestras"])
mcmc2_df = mcmc2_df.drop(columns=["theta"])
mcmc2_df = mcmc2_df.drop(columns=["sigma2_j"])
mcmc2_df = mcmc2_df[1000:]
mcmc2_df.to_csv(PROCESSED_DATA_DIR / "mcmc2_df.csv")

# log-verosimilitud
mcmc2_ll = pd.DataFrame(muestras_mcmc2["LL"])
mcmc2_ll = mcmc2_ll[1000:]
mcmc2_ll.to_csv(PROCESSED_DATA_DIR / "mcmc2_ll.csv")

# %% MCMC3
## Para este punto cambia la notación de ybj y ybk,
## j indica municipios y k indica departamentos
stats_departamento = saber_11.groupby("COD_DEPTO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)

nk = stats_departamento["count"].values  # Número de muestras por departamento
ybk = stats_departamento["mean"].values  # Media de puntaje global por departamento
s2k = stats_departamento["var"].values  # Varianza de puntaje global por departamento


stats_municipio = saber_11.groupby("COD_MCPIO")["PUNT_GLOBAL"].agg(
    ["mean", "var", "count"]
)
nj = stats_municipio["count"].values
ybj = stats_municipio["mean"].values
s2j = stats_municipio["var"].values


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
    m = len(np.unique(cod_depto))
    nk = len(np.unique(cod_mcpio))
    theta = ybk
    zeta = ybj
    kappa2 = np.mean(s2j)
    varsigma2 = np.ones(n)
    sigma2 = np.var(ybj, ddof=1)
    mu = np.mean(theta)
    tau2 = np.var(theta, ddof=1)

    # Indexar zeta_jk
    unique_mcpio = np.unique(cod_mcpio)
    mcpio_to_index = {mcpio: i for i, mcpio in enumerate(unique_mcpio)}
    mcpio_indices = np.vectorize(mcpio_to_index.get)(cod_mcpio)

    # Indexar theta_k
    unique_deptos = np.unique(cod_depto)
    depto_to_index = {depto: i for i, depto in enumerate(unique_deptos)}
    depto_indices = np.vectorize(depto_to_index.get)(cod_depto)

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

    for b in tqdm(range(B), desc="MCMC2 Procesando"):
        # Muestreo de zeta_{j,k}
        np.add.at(sum_inv_varsigma2_jk, mcpio_indices, 1.0 / varsigma2)
        np.add.at(sum_yijk_inv_varsigma2_jk, mcpio_indices, y / varsigma2)
        var_zeta = 1 / (sum_inv_varsigma2_jk + 1 / sigma2)
        mean_zeta = var_zeta * (
            sum_yijk_inv_varsigma2_jk + theta[depto_indices] / sigma2
        )
        zeta = np.random.normal(mean_zeta, np.sqrt(var_zeta))

        # Muestreo de theta_k
        np.add.at(sum_zeta_div_sigma2, depto_indices, zeta / sigma2)
        var_theta = 1 / (np.bincount(depto_indices, weights=1 / sigma2) + 1 / tau2)
        mean_theta = var_theta * (sum_zeta_div_sigma2 + mu / tau2)
        theta = np.random.normal(mean_theta, np.sqrt(var_theta))

        # Muestreo de mu
        var_mu = 1 / (m / tau2 + 1 / gamma2_0)
        mean_mu = var_mu * (np.sum(theta) / tau2 + mu_0 / gamma2_0)
        mu = np.random.normal(mean_mu, np.sqrt(var_mu))

        # Muestreo de tau^2
        eta_n = eta_0 + m
        tau2_n = eta_0 * tau2_0 + np.sum((theta - mu) ** 2)
        tau2 = 1 / np.random.gamma(eta_n / 2, tau2_n / 2)

        # Muestreo de sigma^2
        nu_n = nu_0 + n
        sigma2_n = nu_0 * sigma2_0 + np.sum(
            (zeta[mcpio_indices] - theta[depto_indices]) ** 2
        )
        sigma2 = 1 / np.random.gamma(nu_n / 2, sigma2_n / 2)

        # Muestreo de varsigma^2_{i,j,k}
        extended_zeta = zeta[mcpio_indices]
        kappa2_n = xi_0 * kappa2_0 + np.sum((y - extended_zeta) ** 2)
        varsigma2 = 1 / np.random.gamma((upsilon + 1) / 2, kappa2_n / 2)

        # Muestreo de kappa^2
        xi_n = xi_0 + upsilon * n
        kappa2_n = xi_0 * kappa2_0 + upsilon * np.sum(1 / varsigma2)
        kappa2 = 1 / np.random.gamma(xi_n / 2, kappa2_n / 2)

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
                np.sum(st.norm.logpdf(x=y, loc=extended_zeta, scale=np.sqrt(varsigma2)))
            )

    return {"muestras": muestras, "LL": LL}


# %% Inicializar el muestreador MCMC2
np.random.seed(123)
muestras_mcmc3 = MCMC3(
    B=110000, nj=nj, y=y, cod_depto=cod_depto, s2j=s2j, njk=njk, cod_mcpio=cod_mcpio
)


# %% Data frame de theta
mcmc3_theta = pd.DataFrame(muestras_mcmc3["muestras"]["theta"])
mcmc3_theta = mcmc3_theta[1000:]
mcmc3_theta.to_csv(PROCESSED_DATA_DIR / "mcmc3_theta.csv")

# %% Data frame de sigma2_j
mcmc3_sigma2_j = pd.DataFrame(muestras_mcmc3["muestras"]["sigma2_j"])
mcmc3_sigma2_j
mcmc3_sigma2_j = mcmc3_sigma2_j[1000:]
mcmc3_sigma2_j.to_csv(PROCESSED_DATA_DIR / "mcmc3_sigma2_j.csv")

# Data frame con el resto de parámetros
mcmc3_df = pd.DataFrame(muestras_mcmc3["muestras"])
mcmc3_df = mcmc3_df.drop(columns=["theta"])
mcmc3_df = mcmc3_df.drop(columns=["sigma2_j"])
mcmc3_df = mcmc3_df[1000:]
mcmc3_df.to_csv(PROCESSED_DATA_DIR / "mcmc3_df.csv")

# log-verosimilitud
mcmc3_ll = pd.DataFrame(muestras_mcmc3["LL"])
mcmc3_ll = mcmc3_ll[1000:]
mcmc3_ll.to_csv(PROCESSED_DATA_DIR / "mcmc3_ll.csv")


# %% MCMC4


if __name__ == "__main__":
    MCMC1()
