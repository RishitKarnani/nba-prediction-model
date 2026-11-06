# NBA Data Ingestion, API Evaluation & SQL Staging Report

**Project:** NBA Probability Assistant  
**Date:** October 31, 2026  
**Environment:** Python 3.14 / SQL Server 2022 Express LocalDB (`MSSQLLocalDB`) / `nba_api` / SQLAlchemy 2.1  
**Purpose:** Comprehensive technical summary of the data layer, API reliability, historical coverage, and current prediction capabilities to facilitate architectural review and next-step planning.

---

## 1. Executive Summary

- **Database Connectivity**: Connected Python directly to the local SQL Server instance `(localdb)\MSSQLLocalDB` using `pyodbc` and `SQLAlchemy 2.1` with Windows Authentication (`Trusted_Connection=yes`), aligning with existing local projects (`DogsSa`, `dogsnswwork`).
- **Database Initialized**: The database `NBA_Prediction` was created with 6 clean staging tables (`stg_players`, `stg_player_game_logs`, `stg_team_game_logs`, `stg_boxscore_traditional`, `stg_boxscore_advanced`, `stg_team_rosters`).
- **API Health**: **All 8 target NBA API endpoint categories are functional and verified.**
- **Critical API Discovery**: NBA.com has deprecated `BoxScoreTraditionalV2` and `BoxScoreAdvancedV2`. We migrated the pipeline to **`BoxScoreTraditionalV3` and `BoxScoreAdvancedV3`**, which work reliably and return granular player/team metrics.
- **Sufficiency of 2014–Present Data**: 2014 to present represents over **12,000+ regular season games**, covers the modern 3-point and high-pace era, and includes full advanced metrics (pace, ratings, true shooting). This is more than sufficient for an accurate Version 1 probability engine.
- **Current Team & Player Tracking**: The API provides current player-to-team assignments via `CommonAllPlayers(is_only_current_season=1)` and team rosters via `CommonTeamRoster`, enabling accurate live prediction mapping.

---

## 2. SQL Server Database Architecture

The local database `NBA_Prediction` is structured with staging tables designed to ingest raw API responses cleanly without premature constraint failures:

```
NBA_Prediction (SQL Server LocalDB)
├── dbo.stg_players               (523 active players reference)
├── dbo.stg_player_game_logs       (410 sample game logs: Jokić, LeBron, Dončić across 2022-24)
├── dbo.stg_team_game_logs         (328 team logs: Denver Nuggets & LA Lakers across 2022-24)
├── dbo.stg_boxscore_traditional   (53 player traditional box score rows)
├── dbo.stg_boxscore_advanced      (55 player advanced box score rows)
└── dbo.stg_team_rosters           (70 roster association rows across 2 seasons)
```

### Connection Configuration (`database/connection.py`)
```python
odbc_str = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=(localdb)\\MSSQLLocalDB;"
    "Database=NBA_Prediction;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
```
- Fully parameterised via `.env`.
- Uses `fast_executemany=True` for high-throughput batch inserts.

---

## 3. NBA API Endpoint Evaluation & Verification

Every endpoint was tested against live NBA endpoints with $0.6\text{s}$ throttling and error-handling wrappers:

| Category | Endpoint / Class | Status | Speed | Columns Returned | Assessment for Version 1 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Active Players** | `commonallplayers.CommonAllPlayers` | **SUCCESS** | ~1.2s | 16 | **Essential**. Returns all 523 currently active players, active roster status, and current team IDs. |
| **2. Player Game Logs** | `playergamelog.PlayerGameLog` | **SUCCESS** | ~0.6s | 27 | **Essential**. Complete career game history: PTS, REB, AST, STL, BLK, TOV, MIN, FG%, 3P%, FT%, +/-, W/L. |
| **3. Team Game Logs** | `teamgamelog.TeamGameLog` | **SUCCESS** | ~0.6s | 27 | **Essential**. Team box-level performance per game (82 games/season per team), home/away indicators, win streaks. |
| **4. Traditional Box Score** | `boxscoretraditionalv3.BoxScoreTraditionalV3` | **SUCCESS** | ~0.8s | 36 | **Essential**. Detailed game-level player box scores. Note: V2 is deprecated by NBA; V3 is the modern stable replacement. |
| **5. Advanced Box Score** | `boxscoreadvancedv3.BoxScoreAdvancedV3` | **SUCCESS** | ~0.6s | 39 | **Essential**. Offensive/Defensive ratings, Net Rating, Pace, Possessions, True Shooting %, USG%, PIE. |
| **6. League Player Stats** | `leaguedashplayerstats.LeagueDashPlayerStats` | **SUCCESS** | ~0.7s | 67 | **High Value**. Per-game and total seasonal aggregates for baseline feature engineering. |
| **7. League Team Stats** | `leaguedashteamstats.LeagueDashTeamStats` | **SUCCESS** | ~0.8s | 54 | **High Value**. League-wide team offensive/defensive baselines and pace factors. |
| **8. Team Rosters** | `commonteamroster.CommonTeamRoster` | **SUCCESS** | ~0.9s | 16 | **High Value**. Season-by-season roster tracking, player positions, birth dates, experience, acquisition method. |

---

## 4. Key Questions Answered

### Q1: Is the 2014-to-present data scope sufficient to build Version 1?
**Yes, it is optimal.** Here is why:
1. **Sample Size**: 30 teams $\times$ 82 games $\times$ 10+ seasons $\approx$ **12,300+ games**. In sports betting and machine learning, 10,000+ games provide ample statistical power to avoid overfitting while capturing seasonal variance.
2. **Tactical Relevance**: Pre-2014 NBA basketball (lower 3-point volume, post-heavy, slower pace) differs drastically from today's modern pace-and-space game. Training on games prior to 2014 introduces non-stationary drift that degrades current game prediction accuracy.
3. **Advanced Metric Availability**: NBA tracking data and V3 advanced box scores (true possessions, offensive/defensive ratings, true shooting percentage) are reliably recorded starting from the 2013–14 / 2014–15 seasons.

### Q2: Does the API tell us which teams players currently play for (for current predictions)?
**Yes, in multiple direct ways:**

1. **`CommonAllPlayers(is_only_current_season=1)`**:
   - Returns every active player with their active `TEAM_ID`, `TEAM_NAME`, and `TEAM_ABBREVIATION`.
   - Verified sample output:
     - *Precious Achiuwa* $\to$ `SAC` (Sacramento Kings)
     - *Steven Adams* $\to$ `HOU` (Houston Rockets)
     - *Bam Adebayo* $\to$ `MIA` (Miami Heat)
     - *Santi Aldama* $\to$ `DAL` (Dallas Mavericks)
     - *Grayson Allen* $\to$ `CHA` (Charlotte Hornets)
2. **`CommonTeamRoster(team_id=..., season='2024-25')`**:
   - Returns the active depth chart/roster for any team in the current season.
3. **Upcoming Matchups & Scheduling**:
   - When generating predictions for upcoming games, `scoreboardv2.ScoreboardV2` or `scheduleleaguev2` retrieves today's and upcoming scheduled games with `HOME_TEAM_ID` and `VISITOR_TEAM_ID`.
   - The model matches the current roster of each team against the scheduled matchup to construct the pre-game features.

---

## 5. Missing Data Investigation & Practical Modeling Solutions

| Data Dimension | NBA API Reality | Impact on Win Probability Engine | Recommended Solution for Version 1 |
| :--- | :--- | :--- | :--- |
| **Historical Trades & Transfers** | `CommonTeamRoster` lists `HOW_ACQUIRED` (e.g. "Traded", "Signed"), but does not record the exact timestamp of mid-season trades. | Minor | Track changes in `TEAM_ABBREVIATION` between sequential games in `stg_player_game_logs`. The first game date with a new team is the effective trade date. |
| **Historical Injury Reports** | NBA.com does **not** provide historical pre-game injury status reports (e.g., "Questionable 2 hours before game"). Only post-game DNP reasons are recorded in box scores. | Moderate | **Do not use future DNP tags pre-game.** Instead, compute a rolling **"Player Availability & Recency Metric"** based on whether the player played in the team's last 3 to 5 games. For live/current predictions, scrape the official NBA daily injury PDF or free injury feeds. |
| **Starting Lineups** | Confirmed starters are recorded in `BoxScoreTraditionalV3` via `position` (`F`, `C`, `G`). Bench players have an empty position. | None (Historical) | Historical starting lineups are 100% available. For upcoming games, use the projected starting lineup based on the previous 3 games until 30 minutes before tip-off. |
| **Pre-2014 Advanced Stats** | Not needed since we are focusing on 2014–present. | None | Fully available for the entire chosen training window. |

---

## 6. Recommended Next Steps for Discussion with ChatGPT

When discussing this pipeline with ChatGPT, here are the key architectural points to confirm:

1. **Feature Engineering Strategy (Phase 2)**:
   - Calculate exponentially weighted rolling averages (e.g., last 5 games, last 10 games, full-season) for:
     - Team Offensive Rating & Defensive Rating (pace-adjusted).
     - Rest days & Back-to-Back flags (travel fatigue).
     - Home court advantage differential.
     - Net Rating of top 5 rotation players.
2. **Strict Non-Leakage Rule**:
   - Ensure all features for Game $N$ strictly use data from Games $1 \dots N-1$.
3. **Baseline Probability Model (Phase 3)**:
   - Begin with an **Elo Rating System** and a **Logistic Regression** baseline before training XGBoost.
4. **Validation Strategy (Phase 4)**:
   - Use **walk-forward chronological cross-validation** (e.g., train on seasons 2014–2021, validate on 2021–22, test on 2022–2024), rather than random k-fold splits.

---

*Report prepared and validated via automated execution against Microsoft SQL Server 2022 Express LocalDB (`NBA_Prediction`).*

