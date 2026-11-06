# NBA Prediction Model — Parquet Dataset & Data Dictionary

## 1. Overview
This dataset contains complete historical regular season NBA game logs from the modern high-pace / 3-point era (**2014–15 season through 2026–27 season**), totaling **13 full seasons**, **14,527 unique games**, and over **338,000 game-level records**.

It is formatted entirely in **Apache Parquet** with Snappy compression, preserving exact data types (integers, floats, dates, and strings) for instant, lossless loading in Python, Pandas, Polars, DuckDB, or AI data science workflows.

---

## 2. Directory Structure Inside This Package

```
nba_prediction_dataset/
├── DATA_DICTIONARY.md             # This comprehensive reference & feature engineering guide
├── raw/
│   ├── team_logs/                 # Complete game logs for all 30 NBA teams across 13 seasons
│   │   ├── team_logs_2014-15.parquet
│   │   ├── team_logs_2015-16.parquet
│   │   └── ... (up to 2026-27)
│   └── player_logs/               # Individual player box score logs across 13 seasons
│       ├── player_logs_2014-15.parquet
│       ├── player_logs_2015-16.parquet
│       └── ... (up to 2026-27)
└── sample_api_endpoints/          # Granular investigation parquet samples for all tested NBA endpoints
    ├── boxscoreadvancedv3.parquet     # Game-level Off/Def Rating, Pace, USG%, TS%, PIE
    ├── boxscoretraditionalv3.parquet  # Game-level player box scores (Points, Rebounds, Assists, etc.)
    ├── commonallplayers.parquet       # Complete directory of 523 active players and draft history
    ├── commonteamroster.parquet       # Team roster with player positions (G/F/C), heights, weights
    ├── leaguedashplayerstats.parquet  # Season-level aggregated player stats & league ranks
    ├── leaguedashteamstats.parquet    # Season-level aggregated team stats & league ranks
    ├── playergamelog.parquet          # Sample single-player game log structure
    └── teamgamelog.parquet            # Sample single-team game log structure
```

---

## 3. Team Game Logs (`raw/team_logs/`)
- **Total Rows:** 29,054 records
- **Unit of Observation:** One row per team per game (each game has exactly 2 rows: one for the home team, one for the away team).
- **Primary Key:** `(TEAM_ID, GAME_ID)`

| Column Name | Data Type | Description & Basketball Context |
| :--- | :--- | :--- |
| `SEASON_ID` | string | NBA internal season code (e.g. `'22023'` = Regular Season 2023-24). |
| `TEAM_ID` | int64 | Unique NBA franchise ID (e.g. `1610612743` for Denver Nuggets). |
| `TEAM_ABBREVIATION` | string | Standard 3-letter team code (e.g. `'DEN'`, `'BOS'`, `'LAL'`). |
| `TEAM_NAME` | string | Full team franchise name (e.g. `'Denver Nuggets'`). |
| `GAME_ID` | string | Unique 10-digit NBA game identifier (e.g. `'0022300061'`). |
| `GAME_DATE` | string | Date game was played in `YYYY-MM-DD` format (e.g. `'2023-10-24'`). |
| `MATCHUP` | string | Matchup indicator. **Contains `'vs.'` for Home games** (e.g. `'DEN vs. LAL'`) and **`'@'` for Away games** (e.g. `'DEN @ GSW'`). |
| `WL` | string | Game outcome for this team: `'W'` (Win) or `'L'` (Loss). |
| `MIN` | int64 | Total team minutes played. Standard regulation = 240 (5 players x 48 mins). Overtime = +25 mins per OT. |
| `FGM` | int64 | Total Field Goals Made (2-pointers + 3-pointers). |
| `FGA` | int64 | Total Field Goals Attempted. |
| `FG_PCT` | float64 | Field Goal Percentage (`FGM / FGA`). |
| `FG3M` | int64 | 3-Point Field Goals Made. |
| `FG3A` | int64 | 3-Point Field Goals Attempted. |
| `FG3_PCT` | float64 | 3-Point Field Goal Percentage (`FG3M / FG3A`). |
| `FTM` | int64 | Free Throws Made. |
| `FTA` | int64 | Free Throws Attempted. |
| `FT_PCT` | float64 | Free Throw Percentage (`FTM / FTA`). |
| `OREB` | int64 | Offensive Rebounds (critical for second-chance scoring opportunities). |
| `DREB` | int64 | Defensive Rebounds (critical for ending opponent possessions). |
| `REB` | int64 | Total Rebounds (`OREB + DREB`). |
| `AST` | int64 | Assists (indicates offensive ball movement and shot creation). |
| `STL` | int64 | Steals (live-ball turnovers created leading to high-efficiency transition points). |
| `BLK` | int64 | Blocks (rim protection and interior defensive disruption). |
| `TOV` | int64 | Turnovers committed (empty possessions given up). |
| `PF` | int64 | Personal Fouls committed. |
| `PTS` | int64 | Total points scored in the game by this team. |
| `PLUS_MINUS` | int64 | Final scoring margin (`PTS - Opponent_PTS`). Positive = Win margin; Negative = Loss margin. |
| `VIDEO_AVAILABLE`| int64 | NBA tracking flag (1 = video tracking available, 0 = not). |
| `SEASON` | string | Readable season label (e.g. `'2023-24'`). |
| `downloaded_at` | string | Ingestion pipeline timestamp. |

---

## 4. Player Game Logs (`raw/player_logs/`)
- **Total Rows:** 309,118 records
- **Unit of Observation:** One row per player per game played.
- **Primary Key:** `(PLAYER_ID, GAME_ID)`

| Column Name | Data Type | Description & Basketball Context |
| :--- | :--- | :--- |
| `SEASON_ID` | string | NBA internal season code (e.g. `'22023'`). |
| `PLAYER_ID` | int64 | Unique NBA person ID (e.g. `203999` for Nikola Jokić, `2544` for LeBron James). |
| `PLAYER_NAME` | string | Player full name. |
| `TEAM_ID` | int64 | Team ID the player was on during this specific game. |
| `TEAM_ABBREVIATION` | string | Team 3-letter code. |
| `TEAM_NAME` | string | Team franchise name. |
| `GAME_ID` | string | Unique 10-digit game identifier matching `team_logs`. |
| `GAME_DATE` | string | Game date (`YYYY-MM-DD`). |
| `MATCHUP` | string | Team matchup string (`'DEN vs. LAL'` or `'DEN @ GSW'`). |
| `WL` | string | Game outcome for player's team (`'W'` or `'L'`). |
| `MIN` | int64 | Individual minutes played in this game. (Key for weighting player contributions). |
| `FGM`, `FGA`, `FG_PCT` | int / float | Individual 2P + 3P shooting totals and accuracy. |
| `FG3M`, `FG3A`, `FG3_PCT` | int / float | Individual 3-point shooting volume and efficiency. |
| `FTM`, `FTA`, `FT_PCT` | int / float | Individual free throw shooting totals and accuracy. |
| `OREB`, `DREB`, `REB` | int64 | Individual offensive, defensive, and total rebounds grabbed. |
| `AST` | int64 | Individual assists recorded. |
| `STL` | int64 | Individual steals recorded. |
| `BLK` | int64 | Individual shots blocked. |
| `TOV` | int64 | Individual turnovers committed. |
| `PF` | int64 | Individual personal fouls committed (foul trouble indicator). |
| `PTS` | int64 | Individual points scored. |
| `PLUS_MINUS` | int64 | Net score differential while this specific player was on the floor. |
| `FANTASY_PTS` | float64 | Standard NBA Fantasy Points total for the performance. |
| `VIDEO_AVAILABLE`| int64 | NBA tracking flag. |
| `SEASON` | string | Readable season string (e.g. `'2023-24'`). |
| `downloaded_at` | string | Ingestion pipeline timestamp. |

---

## 5. Sample Endpoint Datasets (`sample_api_endpoints/`)
Included for deep-dive investigation into roster compositions and advanced metrics:
- **`boxscoreadvancedv3.parquet`**: Contains advanced player/team rate stats per game, including `offensiveRating`, `defensiveRating`, `netRating`, `trueShootingPercentage`, `usagePercentage`, `pace`, `possessions`, and `PIE` (Player Impact Estimate).
- **`boxscoretraditionalv3.parquet`**: Detailed modern V3 box score schema directly from the NBA Live Data Feed.
- **`commonallplayers.parquet`**: Reference table of 523 active NBA players, roster statuses, and career span (`FROM_YEAR` to `TO_YEAR`).
- **`commonteamroster.parquet`**: Roster table with physical demographics (`POSITION`, `HEIGHT`, `WEIGHT`, `AGE`, `EXP`, `HOW_ACQUIRED`).
- **`leaguedashplayerstats.parquet` & `leaguedashteamstats.parquet`**: Comprehensive multi-category season ranking tables (points rank, defense rank, etc.).

---

## 6. Quick Start: Loading with Python & Pandas

```python
import pandas as pd
import glob

# 1. Load All 13 Seasons of Team Logs into One DataFrame
df_teams = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("raw/team_logs/*.parquet"))], ignore_index=True)
df_teams["GAME_DATE"] = pd.to_datetime(df_teams["GAME_DATE"])
df_teams = df_teams.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

# 2. Add Home / Away Flag
df_teams["IS_HOME"] = df_teams["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)

# 3. Load All 13 Seasons of Player Logs
df_players = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("raw/player_logs/*.parquet"))], ignore_index=True)
df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"])

print(f"Loaded {len(df_teams):,} team logs and {len(df_players):,} player logs!")
```

---

## 7. High-Signal Feature Engineering Guide for AI

When instructing your AI to generate features from this dataset, point it toward these proven basketball modeling pillars:

### A. Dean Oliver's Four Factors (80% of Game Variance)
1. **Effective Field Goal % (eFG%):**  
   $$\text{eFG\%} = \frac{\text{FGM} + 0.5 \times \text{FG3M}}{\text{FGA}}$$
2. **Turnover Percentage (TOV%):**  
   $$\text{TOV\%} = \frac{\text{TOV}}{\text{FGA} + 0.44 \times \text{FTA} + \text{TOV}}$$
3. **Offensive Rebound Rate (ORB%):**  
   $$\text{ORB\%} = \frac{\text{OREB}}{\text{OREB} + \text{Opponent\_DREB}}$$
4. **Free Throw Rate (FT Rate):**  
   $$\text{FT\_Rate} = \frac{\text{FTM}}{\text{FGA}}$$

### B. Possessions & Pace
$$\text{Possessions} \approx \text{FGA} + 0.44 \times \text{FTA} - \text{OREB} + \text{TOV}$$
$$\text{Offensive Rating (ORtg)} = 100 \times \frac{\text{PTS}}{\text{Possessions}}$$
$$\text{Defensive Rating (DRtg)} = 100 \times \frac{\text{Opponent\_PTS}}{\text{Possessions}}$$
$$\text{Net Rating} = \text{ORtg} - \text{DRtg}$$

### C. Schedule & Fatigue Context
- **Rest Days:** `df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days`
- **Back-to-Back (B2B) Indicator:** Flag when Rest Days $== 1$.
- **3-in-4 Nights:** Flag when a team is playing its 3rd game in 4 calendar days.
- **Rest Advantage Differential:** `Home_Team_Rest_Days - Away_Team_Rest_Days`.

### D. Rolling Form & Momentum (Avoid Data Leakage!)
* **Rule:** Compute rolling statistics (e.g. Exponential Moving Average or last 5 / last 10 games) using `.shift(1)` so the current game's statistics are **never** leaked into the feature vector!
* **Key Rolling Features:** Rolling Net Rating, Rolling eFG%, Rolling Turnover Rate, Rolling Pace differential.

### E. Matchup Differentials (The Input Vector for ML)
For any upcoming matchup between Home Team $H$ and Away Team $A$:
$$\Delta \text{Feature} = \text{Rolling}_{H} - \text{Rolling}_{A}$$
(e.g., $\Delta \text{Net Rating} = \text{Home\_Net\_Rating}_{10} - \text{Away\_Net\_Rating}_{10}$).

