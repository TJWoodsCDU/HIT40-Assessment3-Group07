from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent


CUTOFF = pd.Timestamp("2026-06-11")

# Rules from the brief
N_MATCHES = 104        # Model 2.1: one row per match
N_ROWS_22 = 208        # Model 2.2: one row per team per match
N_FEATURES = 8         # exactly 8 explanatory variables in each model
MAX_SHARED = 4         # at most 4 variables shared between the two models

REST_CAP_DAYS = 10

# Model 2.1 (goal difference): variable name .
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


FORBIDDEN_WORDS = ["goals_a", "goals_b", "goal_diff", "goals_scored", "shots", "xg",
                   "possession", "fouls", "corners", "cards", "attendance"]


def data_dir(mode):
    """mode is 'dummy' or 'real'."""
    return HERE / "data" / mode


def out_dir(mode):
    d = HERE / "outputs" / mode
    d.mkdir(parents=True, exist_ok=True)
    return d