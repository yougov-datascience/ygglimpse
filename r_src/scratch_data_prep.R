
say24 <- say24_6
  
whys <- c("why_moved_to_current_state", "why_think_pid", "why_are_pid", 
"why_think_pid_with_when", "why_are_pid_with_when")

oe_say24_response <- say24 |> select(person_id, starts_with(c("why_","when_", "fraud_"))& !ends_with("_rnd"))

oe_say24_response |> select(person_id, "when_think_pid") |> 
  filter(!str_detect(oe_say24_response$when_think_pid,"__NA__")) |>
  write_csv("~/Documents/GitHub/ygglimpse/py_fodder/when_think_pid_oe.csv")
