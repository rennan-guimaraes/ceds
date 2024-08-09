library(renv)

renv::init()
renv::activate()

data <- read.table("./data/German data file.data", header = FALSE, sep = " ")

summary(data)

install.packages("tidyverse")
library(tidyverse)

