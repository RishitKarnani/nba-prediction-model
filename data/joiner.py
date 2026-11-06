"""
data/joiner.py
──────────────
Joins team logs with their per-game opponent stats.

After joining, every team-game row has:
  - Its own stats (OWN_*)  — untouched original columns
  - Opponent stats (OPP_*) — opponent's stats in the same game

This is the foundation for:
  - Defensive rating calculations
  - Net rating
  - Offensive rebound % (needs opponent DREB)
  - Player-facing opponent defensive context
"""

from __future__ import annotations

import pandas as pd

from data.cleaner import clean_team_logs
from data.loader import get_team_logs


def build_team_game_pairs(team_logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    For each game, join each team row with its opponent's row.

    Returns a DataFrame with 29,054 rows (one per team per game) where:
    - Original columns remain as-is (own team stats)
    - Opponent columns are prefixed with OPP_

    Parameters
    ----------
    team_logs : optional pre-cleaned team logs. If None, loads and cleans automatically.
    """
    if team_logs is None:
        raw = get_team_logs()
        team_logs = clean_team_logs(raw)

    # Self-join on GAME_ID where the two teams are different
    home = team_logs.copy()
    away = team_logs.copy()

    # Columns to bring in from the opponent side
    opp_cols = [
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF",
        "POSSESSIONS", "OFF_RATING", "EFG_PCT", "TOV_PCT", "FT_RATE",
    ]

    # Rename opponent columns with OPP_ prefix
    opp_rename = {c: f"OPP_{c}" for c in opp_cols}
    away_renamed = away[["GAME_ID"] + opp_cols].rename(columns=opp_rename)

    # Merge: each home row gets the other team's stats for that game
    paired = home.merge(
        away_renamed,
        on="GAME_ID",
        how="inner",
        suffixes=("", "_opp_dup"),
    )

    # Remove self-join rows (team vs. itself)
    paired = paired[paired["TEAM_ID"] != paired["OPP_TEAM_ID"]].copy()

    # Defensive Rating = 100 * opponent PTS / own possessions
    paired["DEF_RATING"] = (paired["OPP_PTS"] / paired["POSSESSIONS"]) * 100.0

    # Net Rating
    paired["NET_RATING"] = paired["OFF_RATING"] - paired["DEF_RATING"]

    # Offensive Rebound % = own OREB / (own OREB + opponent DREB)
    oreb_denom = paired["OREB"] + paired["OPP_DREB"]
    paired["OREB_PCT"] = (paired["OREB"] / oreb_denom.clip(lower=1))

    # Win indicator (numeric)
    paired["WIN"] = (paired["WL"] == "W").astype(int)

    return paired.reset_index(drop=True)


def get_opponent_team_id(player_matchup: str, player_team_abbrev: str, team_logs: pd.DataFrame) -> int | None:
    """
    Given a player's MATCHUP string and their TEAM_ABBREVIATION, return the
    opponent's TEAM_ID from team_logs.

    Example: matchup="DEN vs. LAL", team_abbrev="DEN" → LAL team_id
    """
    parts = player_matchup.replace("vs.", "@").split("@")
    if len(parts) != 2:
        return None

    own = player_team_abbrev.strip().upper()
    for part in parts:
        abbrev = part.strip().upper()
        if abbrev and abbrev != own:
            match = team_logs[team_logs["TEAM_ABBREVIATION"].str.upper() == abbrev]
            if not match.empty:
                return int(match.iloc[0]["TEAM_ID"])
    return None
