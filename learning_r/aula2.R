# Ativar o ambiente virtual gerenciado pelo renv
library(renv)
renv::activate()

# Carregar os pacotes necessários (os pacotes já estarão instalados no ambiente gerenciado pelo renv)
library(dplyr)
library(ggplot2)

# Carregar e visualizar os dados
data <- read.csv("./data_aula2/German data file.data")
summary(data)