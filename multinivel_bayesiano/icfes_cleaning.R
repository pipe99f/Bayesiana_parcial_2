library(tidyverse)
library(dplyr)
library(tidyr)
# read data
icfes <- read.csv("Saber 11 2022-2.TXT", sep = ";", dec = ",")
str(icfes)

filtered <- icfes %>%
  select(
    PUNT_GLOBAL,
    COLE_COD_MCPIO_UBICACION,
    COLE_MCPIO_UBICACION,
    COLE_COD_DEPTO_UBICACION,
    COLE_DEPTO_UBICACION,
    ESTU_NACIONALIDAD,
    ESTU_PAIS_RESIDE,
    ESTU_ESTADOINVESTIGACION,
  ) %>%
  rename(COD_MCPIO = COLE_COD_MCPIO_UBICACION, MCPIO = COLE_MCPIO_UBICACION, COD_DEPTO = COLE_COD_DEPTO_UBICACION, DEPTO = COLE_DEPTO_UBICACION) %>%
  drop_na() %>%
  filter(COD_DEPTO != 88) %>%
  filter(ESTU_NACIONALIDAD == "COLOMBIA") %>%
  filter(ESTU_PAIS_RESIDE == "COLOMBIA") %>%
  filter(ESTU_ESTADOINVESTIGACION == "PUBLICAR") %>%
  mutate(COD_MCPIO = str_pad(COD_MCPIO, width = 5, side = "left", pad = "0")) %>%
  mutate(COD_DEPTO = str_pad(COD_DEPTO, width = 2, side = "left", pad = "0")) %>%
  select(PUNT_GLOBAL, COD_MCPIO, MCPIO, COD_DEPTO, DEPTO)

municipios <- filtered %>% distinct(COD_MCPIO, MCPIO)
departamentos <- filtered %>% distinct(COD_DEPTO, DEPTO)

filtered <- filtered %>% select(-MCPIO, -DEPTO)

write.csv(filtered, "saber11.csv", row.names = FALSE)
write.csv(municipios, "municipios.csv", row.names = FALSE)
write.csv(departamentos, "departamentos.csv", row.names = FALSE)
