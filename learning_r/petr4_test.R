# Ativar o ambiente virtual gerenciado pelo renv
library(renv)
renv::activate()

# Carregar os pacotes necessários (os pacotes já estarão instalados no ambiente gerenciado pelo renv)
library(dplyr)
library(ggplot2)

data <- read.csv("./tests/Petrobras PN Historical (1).csv")
summary(data)