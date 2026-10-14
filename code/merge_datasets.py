import sys
import numpy as np
import pandas as pd
import config as C


# ------------------------------------------------------------------ load
def load():
    d = C.DATA_DIR
    matches = pd.read_csv(d / "matches.csv", parse_dates=["date", "prev_wc_date_a", "prev_wc_date_b"])
    teams = pd.read_csv(d / "teams.csv", parse_dates=["market_value_date"])
    form = pd.read_csv(d / "form.csv", parse_dates=["last_match_date"])
    return matches, teams, form


# ------------------------------------------------------------------ builders
# Model 2.1: M has one row per match; columns ending _a / _b hold Team A / Team B info.
def rest_days(M, side):
    """Days since the team's previous match (capped). For a first match, count from its last pre-tournament match."""
    previous = M[f"prev_wc_date_{side}"].fillna(M[f"last_match_date_{side}"])
    return (M["date"] - previous).dt.days.clip(upper=C.REST_CAP_DAYS)   # capped, see config.py

BUILD_21 = {
    "fifa_pts_diff":   lambda M: M.fifa_pts_a - M.fifa_pts_b,
    "recent_ppg_diff": lambda M: M.recent_ppg_a - M.recent_ppg_b,
    "adj_gf_diff":     lambda M: M.adj_gf_a - M.adj_gf_b,
    "adj_ga_diff":     lambda M: M.adj_ga_b - M.adj_ga_a,          # flipped: higher = better for Team A
    "value_ratio_log": lambda M: np.log(M.market_value_eur_m_a / M.market_value_eur_m_b),
    "host_adv":        lambda M: M.host_adv,
    "rest_diff":       lambda M: rest_days(M, "a") - rest_days(M, "b"),
    "top5_share_diff": lambda M: M.top5_share_a - M.top5_share_b,
}

# Model 2.2: L has one row per team per match; t_ = the team we predict, o_ = its opponent.
BUILD_22 = {
    "opp_fifa_pts":   lambda L: L.o_fifa_pts,
    "t_adj_gf":       lambda L: L.t_adj_gf,
    "opp_adj_ga":     lambda L: L.o_adj_ga,
    "t_value_log":    lambda L: np.log(L.t_market_value_eur_m),
    "knockout":       lambda L: L.knockout,
    "final_group_md": lambda L: L.final_group_md,
    "t_forwards":     lambda L: L.t_forwards,
    "t_squad_age":    lambda L: L.t_squad_age,
    "t_top5_share":   lambda L: L.t_top5_share,                    # spare, for the possible swap
}


# ------------------------------------------------------------------ build the datasets
def build(matches, teams, form):
    info = teams.drop(columns="market_value_date").merge(form, on="team", how="outer", validate="1:1")
    info = info.set_index("team")

    # Model 2.1
    M = (matches.merge(info.add_suffix("_a"), left_on="team_a", right_index=True, how="left")
                .merge(info.add_suffix("_b"), left_on="team_b", right_index=True, how="left"))
    d21 = M[["match_id", "team_a", "team_b"]].copy()
    for name in C.FEATURES_21:
        d21[name] = BUILD_21[name](M)
    d21["goal_diff"] = M["goal_diff"]

    # Model 2.2: every match appears twice, once from each team's side
    cols = list(info.columns)
    sides = []
    for t, o in (("a", "b"), ("b", "a")):
        L = matches[["match_id", "date", "knockout", "final_group_md"]].copy()
        L["team"] = matches[f"team_{t}"]
        L["opponent"] = matches[f"team_{o}"]
        L["goals_scored"] = matches[f"goals_{t}"]
        L = L.merge(info.add_prefix("t_"), left_on="team", right_index=True, how="left")
        L = L.merge(info.add_prefix("o_"), left_on="opponent", right_index=True, how="left")
        sides.append(L)
    L = pd.concat(sides, ignore_index=True).sort_values(["match_id", "team"]).reset_index(drop=True)
    d22 = L[["match_id", "team", "opponent"]].copy()
    for name in C.FEATURES_22:
        d22[name] = BUILD_22[name](L)
    d22["goals_scored"] = L["goals_scored"]
    return d21, d22


# ------------------------------------------------------------------ checks
def check(matches, teams, form, d21, d22):
    problems, warnings = [], []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    # --- input tables
    need(matches.match_id.is_unique, "match_id is not unique")
    need(len(matches) == C.N_MATCHES, f"matches.csv has {len(matches)} rows, expected {C.N_MATCHES}")
    need((matches.goal_diff == matches.goals_a - matches.goals_b).all(), "goal_diff != goals_a - goals_b somewhere")
    need(matches.host_adv.isin([-1, 0, 1]).all(), "host_adv must be -1, 0 or 1")
    need(len(teams) == 48 and teams.team.is_unique, f"teams.csv should have 48 unique teams, has {len(teams)}")
    need(len(form) == 48 and form.team.is_unique, f"form.csv should have 48 unique teams, has {len(form)}")
    in_matches = set(matches.team_a) | set(matches.team_b)
    need(in_matches <= set(teams.team), f"teams in matches but not in teams.csv: {sorted(in_matches - set(teams.team))}")
    need(in_matches <= set(form.team), f"teams in matches but not in form.csv: {sorted(in_matches - set(form.team))}")
    if int(matches.knockout.sum()) != 32:
        warnings.append(f"expected 32 knockout matches, found {int(matches.knockout.sum())}")
    if int(matches.final_group_md.sum()) != 24:
        warnings.append(f"expected 24 third-group-match rows, found {int(matches.final_group_md.sum())}")

    # --- cutoff: nothing used as a clue may come from after the tournament started
    need((form.last_match_date < C.CUTOFF).all(), "form.csv has a last_match_date on/after the cutoff 2026-06-11")
    need((teams.market_value_date <= C.CUTOFF).all(), "market values were collected after the cutoff")
    for s in ("a", "b"):
        prev = matches[f"prev_wc_date_{s}"]
        need((prev.isna() | (prev < matches.date)).all(), f"prev_wc_date_{s} is not before the match date")

    # --- dataset shape and content
    need(len(d21) == C.N_MATCHES, f"dataset_2_1 has {len(d21)} rows, expected {C.N_MATCHES}")
    need(len(d22) == C.N_ROWS_22, f"dataset_2_2 has {len(d22)} rows, expected {C.N_ROWS_22}")
    need(len(C.FEATURES_21) == C.N_FEATURES, f"Model 2.1 has {len(C.FEATURES_21)} variables, needs {C.N_FEATURES}")
    need(len(C.FEATURES_22) == C.N_FEATURES, f"Model 2.2 has {len(C.FEATURES_22)} variables, needs {C.N_FEATURES}")
    for label, d, feats in (("2.1", d21, C.FEATURES_21), ("2.2", d22, C.FEATURES_22)):
        X = d[list(feats)]
        need(not X.isna().any().any(), f"Model {label}: blank values in {X.columns[X.isna().any()].tolist()}")
        need(np.isfinite(X.to_numpy(dtype=float)).all(), f"Model {label}: infinite values in the clues")
        bad = [f for f in feats if any(w in f.lower() for w in C.FORBIDDEN_WORDS)]
        need(not bad, f"Model {label}: variable names look like in-match information: {bad}")
        y = d["goal_diff"] if label == "2.1" else d["goals_scored"]
        same = [f for f in feats if np.allclose(d[f], y)]
        need(not same, f"Model {label}: variable identical to the response (leakage): {same}")
        const = [f for f in feats if d[f].nunique() < 2]
        need(not const, f"Model {label}: variable never changes: {const}")

    # --- shared variables (max 4)
    shared_names = sorted(set(C.FEATURES_21) & set(C.FEATURES_22))
    shared_stats = sorted(set(C.FEATURES_21.values()) & set(C.FEATURES_22.values()))
    need(len(shared_stats) <= C.MAX_SHARED,
         f"{len(shared_stats)} underlying statistics are shared (max {C.MAX_SHARED}): {shared_stats}")
    return problems, warnings, shared_names, shared_stats


# ------------------------------------------------------------------ main
def main():
    matches, teams, form = load()
    d21, d22 = build(matches, teams, form)
    problems, warnings, shared_names, shared_stats = check(matches, teams, form, d21, d22)

    print("\n=== Merge checks ===")
    print(f"dataset_2_1: {d21.shape[0]} rows, {len(C.FEATURES_21)} variables")
    print(f"dataset_2_2: {d22.shape[0]} rows, {len(C.FEATURES_22)} variables")
    print(f"Shared by exact name: {len(shared_names)} {shared_names}")
    print(f"Shared underlying statistics (strict reading): {len(shared_stats)} {shared_stats}  (limit {C.MAX_SHARED})")
    for w in warnings:
        print("WARNING:", w)
    if problems:
        print("\nPROBLEMS - fix these before modelling:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("All checks passed.")

    out = C.OUT_DIR
    d21.to_csv(out / "dataset_2_1.csv", index=False)
    d22.to_csv(out / "dataset_2_2.csv", index=False)
    print(f"Saved dataset_2_1.csv and dataset_2_2.csv to {out}")


if __name__ == "__main__":
    main()