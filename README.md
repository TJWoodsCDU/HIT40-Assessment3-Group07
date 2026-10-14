# HIT140 Assessment 3 – Group 07

Predicting FIFA World Cup 2026 match results using only information known before kickoff (Objective 2):
a linear regression for the goal difference (Model 2.1, 104 matches) and one for the goals scored by a team (Model 2.2, 208 team match rows).

## Folder layout

```
code/
    part1.py             Part 1: matches.csv
    part2.py             Part 2: teams.csv
    part3.py             Part 3: form.csv
    config.py            Part 4: variable lists, cutoff date, rules from the brief
    response_eda.py      Part 4: baselines and the shape of the results
    merge_datasets.py    Part 4: builds dataset_2_1.csv and dataset_2_2.csv and checks them
    eda.py               Part 4: correlations, VIF, scatterplots
    models.py            Part 4: linear regression and the competing algorithms
    run_all.py           Part 4: runs the four Part 4 steps in order
data/
    raw/                 source files (FBref, FIFA ranking, Transfermarkt, international results)
    processed/           everything the code produces, including all Part 4 tables and figures
Requirements.txt
```

## How to run

Run every command from the repo folder, in this order:

```
pip install -r Requirements.txt
python code/main.py      # needs the three files   -> datasets, EDA, models
```

If one step fails, the message says what to fix.

## What the Part 4 run saves in data/processed/

| Files | Content |
|---|---|
| `dataset_2_1.csv`, `dataset_2_2.csv` | The two modelling datasets (104 and 208 rows, 8 variables each) |
| `response_summary.csv`, `group_comparisons.csv`, `fig_goals_*.png` | Baselines, goals versus a Poisson distribution, knockout and host comparisons |
| `eda_*.csv`, `fig_eda_*.png` | Correlations, VIF, summary tables, scatterplots, heatmaps |
| `model_comparison_*.csv`, `fig_model_comparison_*.png` | Cross-validated R², RMSE and MAE for every algorithm |
| `ols_coefficients_*.csv`, `ols_summary_*.txt`, `fig_diagnostics_*.png` | The linear regression in detail |
| `predictions_*.csv`, `fig_pred_vs_actual_*.png` | Predictions on matches the model had not seen |

## Decisions to mention in the report

- Every variable comes from before 11 June 2026, the first match. The merge stops if any date breaks this.
- Scores are the score after extra time, with penalty shootouts ignored. Team A is the first-listed team in FBref.
- Squad values come from a Wayback Machine capture of Transfermarkt's World Cup participants page dated 10 June 2026. The FIFA ranking is dated 11 June 2026, the opening day.
- Form is adjusted for opponent strength using FIFA points. Nine team-match rows (opponents not in the FIFA ranking) use the default weight.
- Rest days are capped at 10 days. Uruguay's last match before the tournament was on 31 March, which would otherwise count 76 days of rest.
- The model evaluation uses repeated 5-fold cross validation, keeping the two rows of a match together in Model 2.2.
