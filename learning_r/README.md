# Sales Data Analysis Project

Este projeto realiza uma análise de dados de vendas e visualiza os resultados usando `ggplot2`.

## Pré-requisitos

Certifique-se de ter o R instalado em sua máquina. Para instalar o R, visite [CRAN](https://cran.r-project.org/).

## Configuração do Ambiente

Este projeto utiliza `renv` para gerenciar dependências. Para configurar o ambiente, siga os passos abaixo:

1. Clone o repositório:

   ```sh
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Instale o pacote `renv` se ainda não o tiver:

   ```r
   install.packages("renv")
   ```

3. Restaure o ambiente:
   ```r
   renv::restore()
   ```
4. Instale os pacotes:
   ```r
   renv::snapshot()
   ```

## Executando o Projeto

Depois de configurar o ambiente, você pode executar o script principal. Aqui está um exemplo de como carregar os dados, executar a análise e gerar visualizações:

```r
# Carregar o ambiente virtual
library(renv)
renv::activate()

# Carregar os pacotes necessários
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

print(sales_by_category)

# Criar o gráfico
ggplot(sales_by_category, aes(x = Category, y = TotalSales, fill = Category)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Total de Vendas por Categoria", x = "Categoria", y = "Total de Vendas")
```
