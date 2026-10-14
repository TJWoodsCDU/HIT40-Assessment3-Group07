from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                         # the repo folder: code/ and data/ sit side by side

# Every clue must come from BEFORE this date (the first World Cup match).
CUTOFF = pd.Timestamp("2026-06-11")

# Rules from the brief
N_MATCHES = 104        # Model 2.1: one row per match
N_ROWS_22 = 208        # Model 2.2: one row per team per match
N_FEATURES = 8         # exactly 8 explanatory variables in each model
MAX_SHARED = 4         # at most 4 variables shared between the two models

# Rest days are capped at this many days. Beyond about a week and a half extra rest makes no difference,
# and an uncapped value lets one team (for example a long gap before its first match) dominate the regression.
REST_CAP_DAYS = 10

# Model 2.1 (goal difference): variable name -> the underlying statistic it is built from.
# The statistic tag is only used to count how many statistics the two models share.
FEATURES_21 = {
    "fifa_pts_diff":   "fifa_pts",
    "recent_ppg_diff": "recent_ppg",
    "adj_gf_diff":     "adj_gf",
    "adj_ga_diff":     "adj_ga",
    "value_ratio_log": "market_value",
    "host_adv":        "host",
    "rest_diff":       "rest_days",
    "top5_share_diff": "top5_share",
}

# Model 2.2 (goals scored by one team).
# Possible swap discussed with the team: replace t_forwards and t_squad_age with
# "t_top5_share": "top5_share"  (builder already exists) and a World Cup experience variable.
FEATURES_22 = {
    "opp_fifa_pts":   "fifa_pts",
    "t_adj_gf":       "adj_gf",
    "opp_adj_ga":     "adj_ga",
    "t_value_log":    "market_value",
    "knockout":       "knockout",
    "final_group_md": "final_group_md",
    "t_forwards":     "forwards",
    "t_squad_age":    "squad_age",
}

# Column names that must never be used as clues (they only exist after the match).
FORBIDDEN_WORDS = ["goals_a", "goals_b", "goal_diff", "goals_scored", "shots", "xg",
                   "possession", "fouls", "corners", "cards", "attendance"]


DATA_DIR = ROOT / "data" / "processed"     # inputs from Parts 1-3: matches.csv, teams.csv, form.csv
OUT_DIR = ROOT / "data" / "processed"      # every dataset, table and figure from Part 4 is saved here too
OUT_DIR.mkdir(parents=True, exist_ok=True)