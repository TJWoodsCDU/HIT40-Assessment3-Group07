import pandas as pd
import re
from pathlib import Path

#1. Setting
CUTOFF_DATE = pd.Timestamp("2026-06-11")
}   

#2. Loading input files
teams = pd.read_csv("team_names.csv")
rankings = pd.read_csv("fifa_ranking.csv")
players = pd.read_csv("squads_fbref.csv")
market_values = pd.read_csv("market_values.csv")

# Removing accidental spaces around text. 
for df in [teams, rankings, players, market_values]:
    text_columns = df.select_dtypes(
        include=["object","str"]
    ).columns
    for col in text_columns:
        df[col] = df[col].str.strip()

#Converting market value to int and getting it in millions
def convert_market_value(value):
    # Preserve missing values
    if pd.isna(value):
        return float("nan")

    # Convert to a clean, lowercase string
    value = str(value).strip().lower()
    value = value.replace("€", "").replace(",", "").strip()

    # Extract the numeric amount and optional suffix
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(bn|b|m|mn)?", value)

    if not match:
        return float("nan")

    amount = float(match.group(1))
    suffix = match.group(2)

    # Convert everything to millions of euros
    if suffix in ("bn", "b"):
        amount *= 1000
    elif suffix in ("m", "mn"):
        pass
    return amount


market_values["market_value"] = market_values["market_value"].apply(
    convert_market_value
)

print(market_values[["team", "market_value"]].head(10))

# Standardise team column names
teams = teams.rename(columns={"standard_name": "team"})
market_values = market_values.rename(columns={"team": "team"})

name_mapping_1 = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Democratic Republic of the Congo": "Congo DR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Turkiye": "Türkiye",
    "United States": "USA"
}

name_mapping_2 = {
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "South Korea": "Korea Republic"
}

market_values["team"] = market_values["team"].replace(name_mapping_1)
players["team"] = players["team"].replace(name_mapping_2)

# Convert release_date strings to actual datetime values
rankings["release_date"] = pd.to_datetime(
    rankings["release_date"],
    format="%Y-%m-%d",
    errors="coerce"
)

# Check for missing or invalid dates
if rankings["release_date"].isna().any():
    raise ValueError("Missing or invalid ranking dates.")

# Require ranking dates strictly before 11 June 2026
if (rankings["release_date"] > CUTOFF_DATE).any():
    raise ValueError("Ranking date must be before 11 June 2026.")

print("Ranking dates validated successfully.")

# 3. check the team list 
if "fifa_name" not in teams.columns: #we need a column with team name which in our case is fifa name
    raise ValueError("squad_fbref.csv needs a team column.")

# Team names cannot be blank.
if teams["fifa_name"].isna().any() or (teams["fifa_name"] == "").any():
    raise ValueError("The team list contains blank names.")

#Each national team should only appear once
if teams["fifa_name"].duplicated().any():
    raise ValueError("Team names are duplicated")

#Checking if there are exactly 48 teams.
if len(teams) !=48:
    raise ValueError(f"Expected 48 teams, found {len(teams)}.")

#Convert team column into python set
team_names = set(teams["fifa_name"])

# Set the cutoff date
CUTOFF_DATE = pd.Timestamp("2026-06-11")

# Convert release_date to actual datetime values
rankings["release_date"] = pd.to_datetime(
    rankings["release_date"],
    format="%Y-%m-%d",
    errors="coerce"
)

# Check for invalid or missing dates
if rankings["release_date"].isna().any():
    raise ValueError("Missing or invalid ranking dates.")

# Require dates strictly before 11 June 2026
if (rankings["release_date"] > CUTOFF_DATE).any():
    raise ValueError("Ranking date must be before 11 June 2026.")

print("Ranking dates validated successfully.")
print("Date column type:", rankings["release_date"].dtype)

# 4. check the ranking and market value files
required_ranking_cols = {
    "team", "points", "rank", "release_date" }

required_value_cols = {
   "market_value"}

#stop if required columns are missing. 
if not required_ranking_cols.issubset(rankings.columns):
    raise ValueError("Rankings file is missing required columns.")

if not required_value_cols.issubset(market_values.columns):
    raise ValueError("Market Values file is misisng required columns.")

rankings["points"] = pd.to_numeric(
    rankings["points"], errors="coerce")

market_values["market_value"] = pd.to_numeric(
    market_values["market_value"], errors="coerce"
)

if rankings["points"].isna().any():
    raise ValueError("Missing or invalid FIFA points.")

if (rankings["points"] <= 0).any():
    raise ValueError("FIFA points must be positive.")

if rankings["release_date"].isna().any():
    raise ValueError("Missing or invalid ranking dates.")

if (rankings["release_date"] > CUTOFF_DATE).any():
    raise ValueError("Ranking date must be before 11 June 2026.")

# Market values must be present and cannot be negative.
if market_values["market_value"].isna().any():
    raise ValueError("Missing or invalid market values.")

if (market_values["market_value"] < 0).any():
    raise ValueError("Market values cannot be negative.")

# All 48 teams must use the same FIFA ranking publication date.
if rankings["release_date"].nunique() != 1:
    raise ValueError("Use one common FIFA ranking publication date.")

# 5. Check that all team name match in each file
# Keep only the 48 World Cup teams
rankings = rankings[rankings["team"].isin(team_names)].copy()
print("Number of ranking rows:", len(rankings))
print("Number of unique teams:", rankings["team"].nunique())


def check_team_names(df, file_name):
    found = set(df["team"].dropna())

    missing = team_names - found
    unexpected = found - team_names
    if missing or unexpected:
        raise ValueError(
            f"{file_name}: missing={sorted(missing)},"
            f"unexpected={sorted(unexpected)}"
        )
#compare each input file with the agreed team list.
check_team_names(rankings, "fifa_ranking")
check_team_names(market_values, "market_values.csv")
check_team_names(players, "squads_fbref.csv")

# 6.Validate players information
required_player_cols = {
    "team", "player", "position",
    "birth_date", "club"
}
if not required_player_cols.issubset(players.columns):
    raise ValueError("squads_fbref.csv is missing required columns.")

# Every player record needs a player name, position, birth date and club
for col in [
    "player", "position", "birth_date", "club"
]:
    if players[col].isna().any() or (
        players[col].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"Missing information in {col}.")

# Convert birth dates into a date format pandas can calculate with.
# An invalid date becomes NaT (Not a Time).
players["birth_date"] = pd.to_datetime(
    players["birth_date"],
    format="%d/%m/%Y",
    errors="coerce"
)

if players["birth_date"].isna().any():
    raise ValueError("Some player birth dates are invalid.")

#7.  calculating age of every player
age_days = (CUTOFF_DATE - players["birth_date"]).dt.days

# A negative age means the birth date is after the cutoff.
if (age_days < 0).any():
    raise ValueError("A player has a birth date after the cutoff.")

# Convert days into approximate years.
# 365.2425 accounts for leap years over time.
players["age"] = age_days / 365.2425

#8 Count forwards
players["is_forward"] = (
    players["position"]
    .str.contains("FW", case=False, na=False)
    .astype(int)
)
print("Total forwards:", players["is_forward"].sum())

#9 Summarise player data into team statistics
# Group all player rows by national team.
# Each group represents one national team's squad.
squad = players.groupby("team").agg(
    # Mean of player ages = average squad age.
    squad_age=("age", "mean"),

    # Sum of the forward indicators = total forwards.
    forwards=("is_forward", "sum"),

    # Count player rows = number of records in the squad.
    squad_size=("player", "size"),
).reset_index()

# Check that the player table covers all 48 teams.
if len(squad) != 48:
    raise ValueError(
        f"Expected 48 squads, found {len(squad)}."
    )

# This is a basic check that no team has zero players.
if (squad["squad_size"] < 1).any():
    raise ValueError("At least one team has no players.")

# 10 Merge the team tables

result = (
    # Start with the official list of 48 teams
    teams[["team"]]

    # Attach FIFA ranking points
    .merge(
        rankings[["team", "points"]],
        on="team",
        how="left",
        validate="one_to_one",
    )

    # Attach market values
    .merge(
        market_values[["team", "market_value"]],
        on="team",
        how="left",
        validate="one_to_one",
    )

    # Attach squad statistics
    .merge(
        squad[["team", "squad_age", "forwards"]],
        on="team",
        how="left",
        validate="one_to_one",
    )
)

print(result.head())
print("Number of teams:", len(result))

# 11 Validate the final dataset
required_output_cols = [
    "team",
    "points",
    "market_value",
    "squad_age",
    "forwards",
]

result = result[required_output_cols]

# The output must contain exactly 48 rows.
if len(result) != 48:
    raise ValueError("Final dataset must contain 48 rows.")

# Check every required output cell for missing values.
if result.isna().any().any():
    bad_cols = result.columns[result.isna().any()].tolist()
    raise ValueError(f"Missing values in: {bad_cols}")

# Flag clearly implausible ages for investigation.
# This is a sanity check, not the official age definition.
if not result["squad_age"].between(15, 50).all():
    raise ValueError("Check the squad ages and birth dates.")

# Forward counts must be non-negative whole numbers.
if (result["forwards"] < 0).any():
    raise ValueError("Forward counts cannot be negative.")

if (result["forwards"] % 1 != 0).any():
    raise ValueError("Forward counts must be whole numbers.")

# 12 Format and export the final CSV
output_path = "teams.csv"
result.to_csv(output_path, index=False)

# Print confirmation and a preview of the finished table.
print("SUCCESS: teams.csv created")
print(f"Rows: {len(result)}")
print(f"Columns: {len(result.columns)}")
print(f"Saved to: {output_path}")
print(result.head(10).to_string(index=False))
