import scripts.local_batch_infer

#1------------------------------------------------------------- 
oes1 = pd.read_csv("./py_fodder/why_moved_oe.csv")

df1 = run_batch_df(oes1, "why_moved_to_current_state", args)

df1.to_csv("./r_src/py_results/why_moved.csv", index=False)

#2------------------------------------------------------------- 
oes2 = pd.read_csv("./py_fodder/why_thinkpid_oe.csv")

df2 = run_batch_df(oes2, "why_think_pid",args)

df2.to_csv("./r_src/py_results/why_think_pid.csv", index=False)

#3-------------------------------------------------------------
oes3 = pd.read_csv("./py_fodder/why_arepid_oe.csv")

df3 = run_batch_df(oes3, "why_are_pid",args)

df3.to_csv("./r_src/py_results/why_are_pid.csv", index=False)

#4-------------------------------------------------------------
oes4 = pd.read_csv("./py_fodder/why_thinkpid_withwhen_oe.csv")

df4 = run_batch_df(oes4, "why_think_pid_with_when",args)

df4.to_csv("./r_src/py_results/why_think_pid_with_when.csv", index=False)

#5-------------------------------------------------------------
oes5 = pd.read_csv("./py_fodder/why_arepid_withwhen_oe.csv")

df5 = run_batch_df(oes5, "why_are_pid_with_when",args)

df5.to_csv("./r_src/py_results/why_are_pid_with_when.csv", index=False)

#6-------------------------------------------------------------
oes6 = pd.read_csv("./py_fodder/when_think_pid_oe.csv")

df6 = run_batch_df(oes6, "when_think_pid",args)

df6.to_csv("./r_src/py_results/when_think_pid.csv", index=False)

#7-------------------------------------------------------------
oes7 = pd.read_csv("./py_fodder/when_decided_pid_oe.csv")

df7 = run_batch_df(oes7, "when_decided_pid",args)

df7.to_csv("./r_src/py_results/when_decided_pid.csv", index=False)



out1 = pd.read_csv("./py_fodder/out1.csv")
outdf = glimpse_fulltext(out1, "full_text", args)
