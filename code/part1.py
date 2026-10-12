from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Country of each of the 16 stadiums
# Names are written exactly as they appear in FBref's "Venue" column.
VENUE_COUNTRY = {
    "Estadio Banorte": "Mexico",
    "Estadio Akron": "Mexico",
    "Estadio BBVA": "Mexico",
    "BMO Field": "Canada",
    "BC Place Stadium": "Canada",
    "AT&T Stadium": "USA",
    "SoFi Stadium": "USA",
    "MetLife Stadium": "USA",
    "Mercedes-Benz Stadium": "USA",
    "Gillette Stadium": "USA",
    "Reliant Stadium": "USA",
    "Hard Rock Stadium": "USA",
    "Levi's Stadium": "USA",
    "Lincoln Financial Field": "USA",
    "Lumen Field": "USA",
    "GEHA Field at Arrowhead Stadium": "USA",
}

# Set of the three host nations

HOSTS = {"Mexico", "USA", "Canada"}

# Load raw match list and drop blank rows

raw = pd.read_csv(RAW / "matches_fbref.csv", encoding="utf-8-sig")
raw = raw.dropna(subset=["Home", "Away"]).reset_index(drop=True)

# Get the team names by stripping country code
# Home team has the code at the END ("Mexico mx" -> "Mexico")
# Away team has the code at the START ("za South Africa" -> "South Africa")

home = raw["Home"].str.replace(r"\s+[a-z]{2,3}$", "", regex=True).str.strip()
away = raw["Away"].str.replace(r"^[a-z]{2,3}\s+", "", regex=True).str.strip()

names = pd.read_csv(RAW / "team_names.csv", encoding="utf-8-sig")
fbref_to_standard = dict(zip(names["fbref_name"], names["standard_name"]))

# Create a clean table replaces FBRef name with standard name

df = pd.DataFrame({
    "date": pd.to_datetime(raw["Date"]),
    "stage": raw["Round"],
    "team_a": home.map(fbref_to_standard),
    "team_b": away.map(fbref_to_standard),
    "score_raw": raw["Score"],
    "venue": raw["Venue"].str.replace(r"\s*\(Neutral Site\)", "", regex=True).str.strip(),
    "notes": raw["Notes"].fillna(""),
})

# Stops the script and shows the problem rows if any team name is not found in the standard list
unmatched = df.loc[df["team_a"].isna() | df["team_b"].isna()]
assert unmatched.empty, f"Unmatched team names:\n{unmatched}"

# Split the score into goals for Team A and Team B
# Score is after extra time. Penalty shoot-outs appear in brackets,
# we remove the brackets so penalty shoot-out goals are ignored.

score = df["score_raw"].str.replace(r"\(\d+\)", "", regex=True).str.strip()
goals = score.str.split("–", expand=True).astype(int)
df["goals_a"] = goals[0]
df["goals_b"] = goals[1]

# Calculate goal difference 
# A positive number means Team A scored more goals than Team B, negative means the opposite, and 0 means a draw

df["goal_diff"] = df["goals_a"] - df["goals_b"]

# Sort by kick-off order and give each match an ID from 1 to 104
df = df.sort_values("date", kind="stable").reset_index(drop=True)
df.insert(0, "match_id", range(1, len(df) + 1))

# Knockout: 1 for any round after the group stage, otherwise 0

df["knockout"] = (df["stage"] != "Group stage").astype(int)

# Per-team match order: final_group_md and previous WC match date
# Make a "long" table with one row per team per match, ordered by date.

long = pd.concat([
    df[["match_id", "date", "knockout"]].assign(team=df["team_a"], side="a"),
    df[["match_id", "date", "knockout"]].assign(team=df["team_b"], side="b"),
]).sort_values(["team", "date", "match_id"])

long["team_match_no"] = long.groupby("team").cumcount() + 1
long["prev_wc_date"] = long.groupby("team")["date"].shift(1)
long["is_final_group_md"] = ((long["knockout"] == 0) & (long["team_match_no"] == 3)).astype(int)

wide = long.pivot(index="match_id", columns="side",
                  values=["prev_wc_date", "is_final_group_md"])
df["prev_wc_date_a"] = df["match_id"].map(wide[("prev_wc_date", "a")])
df["prev_wc_date_b"] = df["match_id"].map(wide[("prev_wc_date", "b")])

# In a round-robin group, both teams play their 3rd match at the same time,
# so the flag must be the same for Team A and Team B. Check, then keep one column.
fa = df["match_id"].map(wide[("is_final_group_md", "a")])
fb = df["match_id"].map(wide[("is_final_group_md", "b")])
assert (fa == fb).all(), "final_group_md differs between Team A and Team B"
df["final_group_md"] = fa.astype(int)

# host_adv: +1 if Team A is a host playing in its own country,
# -1 if Team B is, otherwise 0

df["venue_country"] = df["venue"].map(VENUE_COUNTRY)
assert df["venue_country"].notna().all(), "A venue is missing from VENUE_COUNTRY"

a_home = df["team_a"].isin(HOSTS) & (df["team_a"] == df["venue_country"])
b_home = df["team_b"].isin(HOSTS) & (df["team_b"] == df["venue_country"])
df["host_adv"] = a_home.astype(int) - b_home.astype(int)

# Checks

assert len(df) == 104, f"Expected 104 matches, got {len(df)}"
assert df["match_id"].is_unique
assert set(df["team_a"]).union(df["team_b"]) == set(names["standard_name"]), "Team list mismatch"
shootouts = df["notes"].str.contains("penalty shoot-out", case=False)
assert (df.loc[shootouts, "goal_diff"] == 0).all(), "Shoot-out match with goal_diff != 0"
assert df["final_group_md"].sum() == 24, "Expected 24 third-matchday matches (12 groups x 2)"
assert (df.loc[df["knockout"] == 0, "stage"] == "Group stage").all()
assert df["prev_wc_date_a"].isna().sum() + df["prev_wc_date_b"].isna().sum() == 48, \
    "Each of the 48 teams should have exactly one blank prev_wc_date (its first match)"

# Save

columns = ["match_id", "date", "stage", "team_a", "team_b", "goals_a", "goals_b",
           "goal_diff", "knockout", "final_group_md", "host_adv",
           "prev_wc_date_a", "prev_wc_date_b"]
for c in ["date", "prev_wc_date_a", "prev_wc_date_b"]:
    df[c] = df[c].dt.strftime("%Y-%m-%d")      # blanks stay blank

df[columns].to_csv(OUT / "matches.csv", index=False, encoding="utf-8")
print(f"Saved {OUT / 'matches.csv'}: {len(df)} rows, all checks passed.")
print(f"Shoot-outs: {shootouts.sum()} | host_adv counts: {df['host_adv'].value_counts().to_dict()}")
