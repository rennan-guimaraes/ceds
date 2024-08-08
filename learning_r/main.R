# Ativar o ambiente virtual gerenciado pelo renv
library(renv)
renv::activate()

# Carregar os pacotes necessários (os pacotes já estarão instalados no ambiente gerenciado pelo renv)
library(dplyr)
library(ggplot2)

# Carregar e visualizar os dados
sales_data <- read.csv("./sales_data.csv")
head(sales_data)

# Resumo dos dados
summary(sales_data)

# Agrupar e resumir os dados
sales_by_category <- sales_data %>%
    group_by(Category) %>%
    summarise(TotalSales = sum(Price * QuantitySold))

sales_by_category

# Criar o gráfico
ggplot(sales_by_category, aes(x = Category, y = TotalSales, fill = Category)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Total de Vendas por Categoria", x = "Categoria", y = "Total de Vendas")
