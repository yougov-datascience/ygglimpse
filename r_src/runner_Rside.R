library(tidyverse)

d1 <- read_csv("r_src/py_results/why_moved.csv") |>
 mutate(
   question = "why_moved_to_current_state",
   text = why_moved_to_current_state
 ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

d2 <- read_csv("r_src/py_results/why_think_pid.csv") |>
  mutate(
    question = "why_think_pid",
    text = why_think_pid
  ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

d3 <- read_csv("r_src/py_results/when_think_pid.csv") |>
  mutate(
    question = "when_think_pid",
    text = when_think_pid
  ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

d4 <- read_csv("r_src/py_results/when_decided_pid.csv") |>
  mutate(
    question = "when_decided_pid",
    text = when_decided_pid
  ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

d5 <- read_csv("r_src/py_results/why_are_pid.csv") |>
  mutate(
    question = "why_are_pid",
    text = why_are_pid
  ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

d6 <- read_csv("r_src/py_results/why_think_pid_with_when.csv") |>
  mutate(
    question = "why_think_pid_with_when",
    text = why_think_pid_with_when
  ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

d7 <- read_csv("r_src/py_results/why_are_pid_with_when.csv") |>
  mutate(
    question = "why_are_pid_with_when",
    text = why_are_pid_with_when
  ) |>
  select(person_id, question, text, prob_machine_generated, n_tokens, glimpse_criterion)

out <- bind_rows(d1, d2, d3, d4, d5, d6, d7) |>
  arrange(person_id, question) |>
  group_by(person_id) |>
  mutate(
    full_text = text |>
      (\(x) x[!is.na(x) & nzchar(x)])() |>
      unique() |>
      paste(collapse = "   ")
  ) |>
  ungroup()

write_csv(out, "./py_fodder/out1.csv") 
