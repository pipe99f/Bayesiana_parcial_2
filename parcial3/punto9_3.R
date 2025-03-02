library(MASS)
library(ggplot2)
library(dplyr)
set.seed(123)

# Carga de datos
crime <- read.table("http://www2.stat.duke.edu/~pdh10/FCBS/Exercises/crime.dat",
                    header = T)

# OLS
ols_model <- lm(y ~ M + So + Ed + Po1 + Po2 + LF + M.F + Pop + NW + U1
                + U2 + GDP + Ineq + Prob + Time, crime)
ols_model$coefficients 
ols_summary <- as.data.frame(confint(ols_model, level = 0.95)) %>%
  rename(lower = "2.5 %", upper = "97.5 %") %>%
  mutate(mean = coef(ols_model), variable = rownames(.))

# Bayes

bayes_model <- function(datos) {
  n = length(datos$y)
  x <- as.matrix(cbind(1, datos[,2:16]))
  y <- as.matrix(datos$y)
  p <- dim(x)[2]
  
  # Hiperparametros
  nu_0 <- 2
  sigma2_0 <- 1
  g = length(datos$y)
  
  # Modelo - previa g
  a <- (g/(g + 1))*(x %*% solve(t(x) %*% x) %*% t(x))
  ssr_g <- t(y) %*% (diag(n) - a) %*% y
  var_beta <- (g/(g + 1))*solve(t(x) %*% x)
  mean_beta <- var_beta %*% t(x) %*% y
  
  num_sim <- 10000
  sigma2 <- matrix(data = NA, nrow = num_sim, ncol = 1) 
  beta <- matrix(data = NA, nrow = num_sim, ncol = p) 
  
  for (i in 1:num_sim) {
    sigma2[i] <- 1/rgamma(1, (nu_0 + n)/2, (nu_0*sigma2_0 + ssr_g)/2)
    beta[i,] <- c(mvtnorm::rmvnorm(1, mean_beta, sigma2[i]*var_beta))
  }
  colnames(beta) <- c("Intercepto", colnames(datos)[2:16])
  
  return(beta)
}

# Inferencia sobre beta

betas_bayes <- bayes_model(crime) 

beta_summary <- data.frame(
  lower = apply(betas_bayes, 2, quantile, probs = 0.025),
  upper = apply(betas_bayes, 2, quantile, probs = 0.975),
  mean = apply(betas_bayes, 2, mean),
  variable = colnames(betas_bayes)
)

# Comparativo con OLS
ggplot(beta_summary, aes(y = variable, x = mean)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.7) +
  geom_point(color = "blue", size = 3) + 
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2, color = "black") + 
  labs(y = "Coeficiente",
       x = "Intervalo") +
  theme_minimal() # Bayes

ggplot(ols_summary, aes(y = variable, x = mean)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.7) +  # Línea en x = 0
  geom_point(color = "purple", size = 3) +  # Media de cada beta
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2, color = "black") +  # Intervalos de confianza
  labs(y = "Coeficiente",
       x = "Intervalo") +
  theme_minimal() # OLS

# Predicción
# Conjuntos de datos de entrenamiento y de testeo
train <- slice_sample(crime, prop = 0.5)
test <- anti_join(crime, train)
x_test <- as.matrix(cbind(1, test[,2:16]))
y_test <- test$y

# OLS datos entrenamiento
predict_ols <- function(train, x_test) {
  modelo <- lm(y ~ M + So + Ed + Po1 + Po2 + LF + M.F + Pop + NW + U1
               + U2 + GDP + Ineq + Prob + Time, train)
  betas <- as.matrix(modelo$coefficients)
  predic_y <- x_test %*% betas
  
  return(predic_y)
} 

y_ols <- predict_ols(train, x_test)

# Gráfica yols vs ytest
df_ols <- data.frame(y_est = y_ols, y_test = test$y)

ggplot(df_ols, aes(x = y_test, y = y_est)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(x = "Datos testeo",
       y = "Predicciones OLS") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5))

# Error de predicción
mean((y_test - y_ols)^2)

# Regresión bayesiana datos entrenamiento
predict_bayes <- function(train, x_test) {
  betas_sim <- bayes_model(train) 
  betas_est <- as.matrix(apply(betas_sim, 2, mean))
  predic_y <- x_test %*% betas_est
  return(predic_y)
}

y_bayes <- predict_bayes(train, x_test)

# Gráfica ybayes vs ytest
df_by <- data.frame(y_est = y_bayes, y_test = test$y)

ggplot(df_by, aes(x = y_test, y = y_est)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(x = "Datos testeo",
       y = "Predicciones Bayes") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5))

# Error de predicción
mean((y_test - y_bayes)^2)

# Repetición estimaciones de y
rept_sim <- function(datos) {
  train <- slice_sample(datos, prop = 0.5)
  test <- anti_join(datos, train)
  x_test <- as.matrix(cbind(1, test[,2:16]))
  y_test <- test$y
  
  y_ols <- predict_ols(train, x_test)
  y_bayes <- predict_bayes(train, x_test)
  
  return(c(ols_error = mean((y_test - y_ols)^2),
           bayes_error = mean((y_test - y_bayes)^2)))
}

data_repts <- replicate(1000, rept_sim(crime))
