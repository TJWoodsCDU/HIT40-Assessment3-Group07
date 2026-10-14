
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- 1. Settings
ROOT = Path(__file__).resolve().parent.parent          # the repo folder (one level above code/)
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "teams.csv"

CUTOFF_DATE = pd.Timestamp("2026-06-11")               # first World Cup match


MARKET_VALUE_DATE = None

TOP5_FIRST_DIVISION_ONLY = False
TOP5_COUNTRIES = {"eng", "es", "de", "it", "fr"}

if MARKET_VALUE_DATE is None:
    raise ValueError("Set MARKET_VALUE_DATE near the top of this file to the real date the values were taken.")
if MARKET_VALUE_DATE > CUTOFF_DATE:
    raise ValueError("Market values must be from on or before 11 June 2026, otherwise they leak the results.")

# ---------------------------------------------------------------- 2. Loading input files
teams = pd.read_csv(RAW / "team_names.csv")
rankings = pd.read_csv(RAW / "fifa_ranking.csv")
players = pd.read_csv(RAW / "squads_fbref.csv")
market_values = pd.read_csv(RAW / "market_values.csv", encoding="utf-8-sig")   # utf-8-sig removes the hidden BOM

# Remove accidental spaces around text (works on pandas 2.x and 3.x)
for df in [teams, rankings, players, market_values]:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()


# Convert market values such as "€1.56bn" or "€946.00m" into millions of euros
def convert_market_value(value):
    if pd.isna(value):
        return float("nan")
    value = str(value).strip().lower().replace("€", "").replace(",", "").strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(bn|b|m|mn)?", value)
    if not match:
        return float("nan")
    amount = float(match.group(1))
    if match.group(2) in ("bn", "b"):
        amount *= 1000
    return amount


market_values["market_value"] = market_values["market_value"].apply(convert_market_value)

# Standardise team names so every file uses the same spelling
teams = teams.rename(columns={"standard_name": "team"})

name_mapping_1 = {                                     # market_values.csv -> FIFA spelling
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Democratic Republic of the Congo": "Congo DR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Turkiye": "Türkiye",
    "United States": "USA",
}
name_mapping_2 = {                                     # squads_fbref.csv -> FIFA spelling
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "South Korea": "Korea Republic",
}
market_values["team"] = market_values["team"].replace(name_mapping_1)
players["team"] = players["team"].replace(name_mapping_2)

# ---------------------------------------------------------------- 3. Check the team list
if "fifa_name" not in teams.columns:
    raise ValueError("team_names.csv needs a fifa_name column.")
if teams["fifa_name"].isna().any() or (teams["fifa_name"] == "").any():
    raise ValueError("The team list contains blank names.")
if teams["fifa_name"].duplicated().any():
    raise ValueError("Team names are duplicated.")
if len(teams) != 48:
    raise ValueError(f"Expected 48 teams, found {len(teams)}.")
if not (teams["team"] == teams["fifa_name"]).all():
    raise ValueError("standard_name and fifa_name differ for some teams; the merges below need them equal.")
team_names = set(teams["fifa_name"])

# ---------------------------------------------------------------- 4. Check the ranking and market value files
for col in ["team", "points", "rank", "release_date"]:
    if col not in rankings.columns:
        raise ValueError(f"fifa_ranking.csv is missing column {col}.")
if "market_value" not in market_values.columns:
    raise ValueError("market_values.csv is missing the market_value column.")

rankings["release_date"] = pd.to_datetime(rankings["release_date"], format="%Y-%m-%d", errors="coerce")
rankings["points"] = pd.to_numeric(rankings["points"], errors="coerce")

if rankings["release_date"].isna().any():
    raise ValueError("Missing or invalid ranking dates.")
if (rankings["release_date"] > CUTOFF_DATE).any():          # on or before the first match day is allowed
    raise ValueError("Ranking date must be on or before 11 June 2026.")
if rankings["release_date"].nunique() != 1:
    raise ValueError("Use one common FIFA ranking publication date.")
if rankings["points"].isna().any() or (rankings["points"] <= 0).any():
    raise ValueError("Missing, invalid or non-positive FIFA points.")
if market_values["market_value"].isna().any() or (market_values["market_value"] < 0).any():
    raise ValueError("Missing, invalid or negative market values.")
print("Ranking date:", rankings["release_date"].iloc[0].date(), "- validated.")

# ---------------------------------------------------------------- 5. Check that team names match in each file
rankings = rankings[rankings["team"].isin(team_names)].copy()


def check_team_names(df, file_name):
    found = set(df["team"].dropna())
    missing, unexpected = team_names - found, found - team_names
    if missing or unexpected:
        raise ValueError(f"{file_name}: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


check_team_names(rankings, "fifa_ranking.csv")
check_team_names(market_values, "market_values.csv")
check_team_names(players, "squads_fbref.csv")

# ---------------------------------------------------------------- 6. Validate the players table
for col in ["team", "player", "position", "birth_date", "club"]:
    if col not in players.columns:
        raise ValueError(f"squads_fbref.csv is missing column {col}.")
    if players[col].isna().any() or (players[col].astype(str).str.strip() == "").any():
        raise ValueError(f"Missing information in {col}.")

players["birth_date"] = pd.to_datetime(players["birth_date"], format="%d/%m/%Y", errors="coerce")
if players["birth_date"].isna().any():
    raise ValueError("Some player birth dates are invalid.")

# ---------------------------------------------------------------- 7. Age of every player on the cutoff date
age_days = (CUTOFF_DATE - players["birth_date"]).dt.days
if (age_days < 0).any():
    raise ValueError("A player has a birth date after the cutoff.")
players["age"] = age_days / 365.2425

# ---------------------------------------------------------------- 8. Forwards (a player can have two positions, e.g. "FW,MF")
players["is_forward"] = players["position"].str.contains("FW", case=False, na=False).astype(int)

# ---------------------------------------------------------------- 9. Share of the squad at top-5-league clubs
# The club column looks like "1.eng West Ham": division number, country code, club name.
club_parts = players["club"].str.extract(r"^(\d+)\.([a-z]+)\s")
if club_parts.isna().any().any():
    raise ValueError("Some club entries do not look like '1.eng Club Name'.")
players["club_division"] = club_parts[0].astype(int)
players["club_country"] = club_parts[1]
players["is_top5"] = players["club_country"].isin(TOP5_COUNTRIES)
if TOP5_FIRST_DIVISION_ONLY:
    players["is_top5"] = players["is_top5"] & (players["club_division"] == 1)
players["is_top5"] = players["is_top5"].astype(int)

# ---------------------------------------------------------------- 10. One row per team
squad = players.groupby("team").agg(
    squad_age=("age", "mean"),
    forwards=("is_forward", "sum"),
    top5_share=("is_top5", "mean"),
    squad_size=("player", "size"),
).reset_index()

if len(squad) != 48:
    raise ValueError(f"Expected 48 squads, found {len(squad)}.")
if (squad["squad_size"] < 1).any():
    raise ValueError("At least one team has no players.")

result = (
    teams[["team"]]
    .merge(rankings[["team", "points"]], on="team", how="left", validate="one_to_one")
    .merge(market_values[["team", "market_value"]], on="team", how="left", validate="one_to_one")
    .merge(squad[["team", "squad_age", "top5_share", "forwards"]], on="team", how="left", validate="one_to_one")
)

# Rename to the agreed output format
result = result.rename(columns={"points": "fifa_pts", "market_value": "market_value_eur_m"})
result["market_value_date"] = MARKET_VALUE_DATE.strftime("%Y-%m-%d")
result["squad_age"] = result["squad_age"].round(3)
result["top5_share"] = result["top5_share"].round(4)
result = result[["team", "fifa_pts", "market_value_eur_m", "market_value_date",
                 "squad_age", "top5_share", "forwards"]]

# ---------------------------------------------------------------- 11. Validate the final table
if len(result) != 48 or result["team"].duplicated().any():
    raise ValueError("Final dataset must contain 48 different teams.")
if result.isna().any().any():
    raise ValueError(f"Missing values in: {result.columns[result.isna().any()].tolist()}")
if not result["squad_age"].between(15, 50).all():
    raise ValueError("Check the squad ages and birth dates.")
if not result["top5_share"].between(0, 1).all():
    raise ValueError("top5_share must be between 0 and 1.")
if (result["forwards"] < 0).any() or (result["forwards"] % 1 != 0).any():
    raise ValueError("Forward counts must be non-negative whole numbers.")

# ---------------------------------------------------------------- 12. Export
OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT, index=False)
print("SUCCESS: teams.csv created")
print(f"Rows: {len(result)}  Columns: {len(result.columns)}  Saved to: {OUT}")
print(result.head(10).to_string(index=False))