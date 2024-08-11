library(renv)

renv::init()
renv::activate()

install.packages("tidyverse")
library(tidyverse)

data <- read.table("./data/German data.data", header = FALSE, sep = " ")

data |> 
  glimpse()

