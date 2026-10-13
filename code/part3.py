import pandas as pd
import numpy as np

# Load match results and FIFA rankings
results = pd.read_csv('data/raw/martj42_international_results.csv')  # from martj42/international_results.git results.csv
fifa_rankings = pd.read_csv('data/raw/fifa_ranking.csv')

# Set date limits and filter teams
results['date'] = pd.to_datetime(results['date'])
cutoff_start = pd.to_datetime('2023-09-01')  # first day of WC2026 qualifiers
cutoff_end = pd.to_datetime('2026-06-10')  # day before WC2026

df = results[(results['date'] >= cutoff_start) & (results['date'] <= cutoff_end)].copy()

# Get standardised names for the 48 teams (unique)
matches_df = pd.read_csv('data/processed/matches.csv')

teams_a = matches_df['team_a'].unique()
teams_b = matches_df['team_b'].unique()

wc_48_team_list = sorted(list(set(teams_a).union(set(teams_b))))

name_mapping = {
    # martj42 name -> standardised 48-team name
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "United States": "USA",

    #opponnents outside the 48

    "China": "China PR",
    "North Korea": "DPR Korea",
    "Gambia": "The Gambia",
    "Kyrgyzstan": "Kyrgyz Republic",
    "saint Lucia": "St. Lucia"
}

# Apply mapping
df['home_team'] = df['home_team'].replace(name_mapping)
df['away_team'] = df['away_team'].replace(name_mapping)


# Average FIFA points across all ranked teams globally (for weighting)
avg_fifa_pts = fifa_rankings['points'].mean()

# Store compiled stats per team
form_data = []

# loop through each of the 48 standard teams
for team in wc_48_team_list:
    # Get all matches involving this team
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date', ascending=False)

    if len(team_matches) == 0:
        print(f"Warning: No matches found for {team}")
        continue

    # Date of last match played before WC
    last_match_date = team_matches['date'].iloc[0].strftime('%Y-%m-%d')
    n_matches = len(team_matches)

    # Recent points per game (PPG)
    recent_5 = team_matches.head(5)
    points = []
    for _, match in recent_5.iterrows():
        is_home = (match['home_team'] == team)
        team_score = match['home_score'] if is_home else match['away_score']
        opp_score = match['away_score'] if is_home else match['home_score']

        if team_score > opp_score: points.append(3)
        elif team_score == opp_score: points.append(1)
        else: points.append(0)

    recent_ppg = np.mean(points) if len(points) > 0 else 0.0

    # Adjusted GF and GA
    adj_gf_list, adj_ga_list = [], []
    for _, match in team_matches.iterrows():
        is_home = (match['home_team'] == team)
        opp_name = match['away_team'] if is_home else match['home_team']

        gf = match['home_score'] if is_home else match['away_score']
        ga = match['away_score'] if is_home else match['home_score']

        # Lookup opponent FIFA points
        opp_pts = fifa_rankings.loc[fifa_rankings['team'] == opp_name, 'points'].values
        opp_fifa_pts = opp_pts[0] if len(opp_pts) > 0 else avg_fifa_pts

        # Weighted goals calculation
        adj_gf_list.append(gf * (opp_fifa_pts / avg_fifa_pts))
        adj_ga_list.append(ga * (avg_fifa_pts / opp_fifa_pts))

    adj_gf = np.mean(adj_gf_list)
    adj_ga = np.mean(adj_ga_list)

    form_data.append({
        'team': team,
        'adj_gf': round(adj_gf, 4),
        'adj_ga': round(adj_ga, 4),
        'recent_ppg': round(recent_ppg, 4),
        'last_match_date': last_match_date,
        'n_matches_used': n_matches
    })

form_df = pd.DataFrame(form_data)
form_df.to_csv('data/processed/form.csv', index=False)
print("form.csv successfully created.")
