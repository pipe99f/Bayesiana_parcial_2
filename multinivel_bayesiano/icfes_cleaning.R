library(dplyr)
library(tidyr)
# read data
icfes <- read.csv("data/raw/Saber 11 2022-2.TXT", sep = ";", dec = ",")
str(icfes)

filtered <- icfes %>%
  select(PUNT_GLOBAL, COLE_COD_MCPIO_UBICACION, COLE_MCPIO_UBICACION, COLE_COD_DEPTO_UBICACION, COLE_DEPTO_UBICACION) %>%
  rename(COD_MCPIO = COLE_COD_MCPIO_UBICACION, MCPIO = COLE_MCPIO_UBICACION, COD_DEPTO = COLE_COD_DEPTO_UBICACION, DEPTO = COLE_DEPTO_UBICACION) %>%
  drop_na() %>%
  filter(COD_DEPTO != 88)

municipios <- filtered %>% distinct(COD_MCPIO, MCPIO)
departamentos <- filtered %>% distinct(COD_DEPTO, DEPTO)

filtered <- filtered %>% select(-MCPIO, -DEPTO)

write.csv(filtered, "saber11.csv", row.names = FALSE)
write.csv(municipios, "municipios.csv", row.names = FALSE)
write.csv(departamentos, "departamentos.csv", row.names = FALSE)
