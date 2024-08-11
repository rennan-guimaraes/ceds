library(renv)

renv::init()
renv::activate()

install.packages("tidyverse")
library(tidyverse)

install.packages("styler")
library(styler)
data <- read.table("./data/German data.data", header = FALSE, sep = " ")

# visualize data
data |>
  glimpse()

# get all columns name
data |>
  colnames()

# rename colums
german_bank_data <- data |>
  rename("account_status" = V1, duration_month = V2,
         credit_history = V3, purpose = V4,
         credit_amount = V5, saving_account = V6,
         employment_since = V7, installment_rate = V8,
         personal_status = V9, other_debtors = V10,
         residence_since = V11, property = V12,
         age = V13, other_installment = V14,
         housing = V15, existing_credits = V16,
         job = V17, people_liable = V18, telephone = V19,
         foreign_worker = V20, good_loan = V21)

# understand account_status
german_bank_data |>
  select(account_status) |>
  distinct()

# transform account_status
german_bank_data <- german_bank_data |>
  mutate(account_status = case_when(
    account_status == "A11" ~ "... <    0 DM",
    account_status == "A12" ~ "0 <= ... <  200 DM",
    account_status == "A13" ~ "... >= 200 DM /
		     salary assignments for at least 1 year",
    account_status == "A14" ~ "no checking account",
  ))

german_bank_data <- german_bank_data |>
  arrange(account_status, .desc = FALSE)

# understand credit_history
german_bank_data |>
  select(credit_history) |>
  distinct()

# transform credit_history
german_bank_data <- german_bank_data |>
  mutate(credit_history = case_when(
    credit_history == "A30" ~ "no credits taken/all credits paid back duly",
    credit_history == "A31" ~ "all credits at this bank paid back duly",
    credit_history == "A32" ~ "existing credits paid back duly till now",
    credit_history == "A33" ~ "delay in paying off in the past",
    credit_history == "A34" ~ "critical account/other credits existing 
    (not at this bank)",
  ))

# understand credit_history
german_bank_data |>
  select(purpose) |>
  distinct()

# transform propose
german_bank_data <- german_bank_data |>
  mutate(purpose = case_when(
    purpose == "A40" ~ "car (new)",
    purpose == "A41" ~ "car (used)",
    purpose == "A42" ~ "furniture/equipment",
    purpose == "A43" ~ "radio/television",
    purpose == "A44" ~ "domestic appliances",
    purpose == "A45" ~ "repairs",
    purpose == "A46" ~ "education",
    purpose == "A47" ~ "(vacation - does not exist?)",
    purpose == "A48" ~ "retraining",
    purpose == "A49" ~ "business",
    purpose == "A410" ~ "others",
  ))

# understand saving_account
german_bank_data |>
  select(saving_account) |>
  distinct() |>
  arrange(saving_account, .desc = FALSE)

# transform saving_account
german_bank_data <- german_bank_data |>
  mutate(saving_account = case_when(
    saving_account == "A61" ~ "... <  100 DM",
    saving_account == "A62" ~ "100 <= ... <  500 DM",
    saving_account == "A63" ~ "500 <= ... < 1000 DM",
    saving_account == "A64" ~ "... >= 1000 DM",
    saving_account == "A65" ~ "unknown/ no savings account",
  ))

# understand employment_since
german_bank_data |>
  select(employment_since) |>
  distinct() |>
  arrange(employment_since, .desc = FALSE)

# transform employment_since
german_bank_data <- german_bank_data |>
  mutate(employment_since = case_when(
    employment_since == "A71" ~ "unemployed",
    employment_since == "A72" ~ "... < 1 year",
    employment_since == "A73" ~ "1  <= ... < 4 years",
    employment_since == "A74" ~ "4  <= ... < 7 years",
    employment_since == "A75" ~ "... >= 7 years",
  ))

# understand installment rate
german_bank_data |>
  select(installment_rate) |>
  distinct() |>
  arrange(installment_rate, .desc = FALSE)

# transform installment rate like professor said
german_bank_data <- german_bank_data |>
  mutate(installment_rate = case_when(
    installment_rate == 1 ~ 0,
    installment_rate == 2 ~ 0.33,
    installment_rate == 3 ~ 0.66,
    installment_rate == 4 ~ 1,
  ))

# understand personal status and sex
german_bank_data |>
  select(personal_status) |>
  distinct() |>
  arrange(personal_status, .desc = FALSE)

# transform personal status and sex
german_bank_data <- german_bank_data |>
  mutate(personal_status = case_when(
    personal_status == "A91" ~ "male   : divorced/separated",
    personal_status == "A92" ~ "female : divorced/separated/married",
    personal_status == "A93" ~ "male   : single",
    personal_status == "A94" ~ "male   : married/widowed",
    personal_status == "A95" ~ "female : single",
    # I didn't find A95 but i categoryze like documentation
  ))

# understand other debtors/gyarantos
german_bank_data |>
  select(other_debtors) |>
  distinct() |>
  arrange(other_debtors, .desc = FALSE)

# transform personal status and sex
german_bank_data <- german_bank_data |>
  mutate(other_debtors = case_when(
    other_debtors == "A101" ~ "none",
    other_debtors == "A102" ~ "co-applicant",
    other_debtors == "A103" ~ "guarantor",
  ))

# understand property
german_bank_data |>
  select(property) |>
  distinct() |>
  arrange(property, .desc = FALSE)

# transform property
german_bank_data <- german_bank_data |>
  mutate(property = case_when(
    property == "A121" ~ "real estate",
    property == "A122" ~ "building society savings agreement/life insurance",
    property == "A123" ~ "car or other",
    property == "A124" ~ "unknown / no property",
  ))

# understand other installment plans
german_bank_data |>
  select(other_installment) |>
  distinct() |>
  arrange(other_installment, .desc = FALSE)

# transform other installment plans
german_bank_data <- german_bank_data |>
  mutate(other_installment = case_when(
    other_installment == "A141" ~ "bank",
    other_installment == "A142" ~ "stores",
    other_installment == "A143" ~ "none",
  ))

# understand housing
german_bank_data |>
  select(housing) |>
  distinct() |>
  arrange(housing, .desc = FALSE)

# transform housing
german_bank_data <- german_bank_data |>
  mutate(housing = case_when(
    housing == "A151" ~ "rent",
    housing == "A152" ~ "own",
    housing == "A153" ~ "for free",
  ))

# understand job
german_bank_data |>
  select(job) |>
  distinct() |>
  arrange(job, .desc = FALSE)

# transform job
german_bank_data <- german_bank_data |>
  mutate(job = case_when(
    job == "A171" ~ "unemployed/ unskilled - non-resident",
    job == "A172" ~ "unskilled - resident",
    job == "A173" ~ "skilled employee/ official",
    job == "A174" ~ "management/ self-employed/ 
    highly qualified employee/ officer",
  ))

# understand telephone
german_bank_data |>
  select(telephone) |>
  distinct() |>
  arrange(telephone, .desc = FALSE)

# transform telephone
german_bank_data <- german_bank_data |>
  mutate(telephone = case_when(
    telephone == "A191" ~ "none",
    telephone == "A192" ~ "yes",
  ))

# understand foreign worker
german_bank_data |>
  select(foreign_worker) |>
  distinct() |>
  arrange(foreign_worker, .desc = FALSE)

# transform foreign worker
german_bank_data <- german_bank_data |>
  mutate(foreign_worker = case_when(
    foreign_worker == "A201" ~ "yes",
    foreign_worker == "A202" ~ "no",
  ))

# understand credit_risk
german_bank_data |>
  select(good_loan) |>
  distinct() |>
  arrange(good_loan, .desc = FALSE)

# transform good_loan
german_bank_data <- german_bank_data |>
  mutate(good_loan = case_when(
    good_loan == 1 ~ "yes",
    good_loan == 2 ~ "no",
  ))

# visualize data
german_bank_data |>
  glimpse()

# save data
write.csv(german_bank_data, "./data/german_bank_data.csv", row.names = FALSE)
