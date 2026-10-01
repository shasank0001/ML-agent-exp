# DataLab — your ML co-pilot

Upload a **CSV** (or Excel/Parquet) with a column you want to predict, then just ask for it.

## Try this first

> Build the best model to predict `churned`

That single sentence runs the whole path: the agent profiles your data, confirms the
target if it is ambiguous, writes a visible plan, runs a baseline, trains a few more
model families, and shows you a results table it can defend.

## Other things you can ask

You are not locked into that pipeline. The agent has a real Python workspace, so
almost anything works:

- *Plot the distribution of `age`.*
- *Drop outliers in `price` and retry the best model.*
- *Show feature importance for the best model.*
- *Why did the random forest beat logistic regression?* — answered from the experiments it actually ran
- *How many missing values are there, and does it look like MCAR?*
- *Cluster the customers into 4 groups and describe each one.*
- *Write a short report of everything we found to `report.md`.*

## What you can see

Every action is visible: the code it wrote, what it printed, the plots, the to-do list, and
any experiment it logged. Metrics are computed by the harness, not by the model, so the
numbers in the results table are the numbers on disk in `runs/<session>/state.json`.

## Things that will ask you first

The agent's code runs **on this machine, with no sandbox**. Before it runs something slow
or destructive — a big grid search, deleting a file, overwriting your uploaded data — it
stops and asks. Approve or deny; denying is a perfectly normal answer and the agent adapts.

## Where your files go

`runs/<session id>/` holds `data/` (your uploads), `outputs/` (files it wrote), `figures/`
(plots), `state.json` (the research record) and `events.jsonl` (the full transcript).
Nothing leaves your machine unless you configured a hosted model provider.
