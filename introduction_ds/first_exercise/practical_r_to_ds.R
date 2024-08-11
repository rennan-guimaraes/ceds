library(renv)

renv::init()
renv::activate()

install.packages("tidyverse")
library(tidyverse)

data <- read.table("./data/German data file.data", header = FALSE, sep = " ")

# ways to undertand my data
View(data)
glimpse(data)
summary(data)

# filter data
data |>
  filter(V1 == "A14")

data |>
  filter(V1 %in% c("A14", "A12"))

# order
data |>
  arrange(V1)

data |>
  arrange(desc(V1))

# remove duplicates
data |>
  distinct()

data |>
  distinct(V1, V2)

data |>
  distinct(V1, V2, .keep_all = TRUE)

# count
data |>
  count(V1)

# create new columns
data |>
  mutate(V3 = paste(V1, V2), .keep="used") |>
  summary()

# select columns
data |>
  select(V1, V2) |>
  summary()

data |>
  select(V1, ends_with("1"))

# rename columns
data |>
  rename(col1 = V1, col2 = V2) |>
  summary()

# group and summarise
data |>
  group_by(V1) |>
  summarise(mean = mean(V2))

data |>
  group_by(V1) |>
  summarise(mean = mean(V2), na.rm = TRUE, n = n())

# slice
df |> slice_head(n = 1)   # pega a primeira linha de cada grupo.
df |> slice_tail(n = 1)   # ocupa a última fileira de cada grupo.
df |> slice_min(x, n = 1) # pega a linha com o menor valor da coluna x.
df |> slice_max(x, n = 1) # pega a linha com o maior valor da coluna x.
df |> slice_sample(n = 1) # pega uma linha aleatória.

