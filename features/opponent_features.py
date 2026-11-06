"""
features/opponent_features.py
──────────────────────────────
Computes opponent-specific defensive features that describe how well a given
opponent defends against particular stat categories.

These features are joined onto the player feature matrix to give the ML model
context about the upcoming or historical opponent.

Key features:
  OPP_DEF_PTS_ALLOWED_MEAN{W}  : rolling avg points allowed to all opponents
  OPP_DEF_{STAT}_ALLOWED_MEAN  : rolling avg of a specific stat allowed to
                                  players at the target position (approximated)
  OPP_NET_RATING_ROLL{W}       : opponent rolling net rating
  OPP_PACE_ROLL{W}             : opponent rolling pace (affects counting stats)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import settings


def build_opponent_defensive_features(
    player_logs: pd.DataFrame,
    team_paired: pd.DataFrame,
    stat: str,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    For each player game, compute how many points/rebounds/assists that
    opponent has allowed to all players in their last W games.

    This is the key signal for 'opponent defensive context'.

    Parameters
    ----------
    player_logs  : cleaned player logs with GAME_ID, TEAM_ID, OPPONENT_ABBREV, stat
    team_paired  : output of joiner.build_team_game_pairs() with OPP_TEAM_ID and ratings
    stat         : the stat to compute opponent allowed averages for
    windows      : rolling window sizes

    Returns
    -------
    DataFrame indexed like player_logs with new OPP_ALLOWED_{stat}_ROLL{W} columns.
    """
    if windows is None:
        windows = settings.ROLLING_WINDOWS

    if stat not in player_logs.columns:
        raise ValueError(f"Stat '{stat}' not found in player_logs columns.")

    out = player_logs.copy()

    # Step 1: For each team-game, sum up how many {stat} all opponents scored against them
    # Group by: the team that was DEFENDING (OPP_TEAM_ID in player_logs terms = player's opponent)
    # We need: for each game, what was the total {stat} the opposing team allowed?

    active = out[out["MIN"] >= settings.MIN_MINUTES_THRESHOLD].copy()

    # Aggregate per game per opponent team: total {stat} conceded
    game_opp_totals = (
        active
        .groupby(["GAME_ID", "OPPONENT_ABBREV"])[stat]
        .sum()
        .reset_index()
        .rename(columns={stat: f"OPP_CONCEDED_{stat}"})
    )

    # Map opponent abbreviation → TEAM_ID for joining
    abbrev_to_id = (
        player_logs[["TEAM_ID", "TEAM_ABBREVIATION"]]
        .drop_duplicates()
        .rename(columns={"TEAM_ABBREVIATION": "OPPONENT_ABBREV", "TEAM_ID": "OPP_TEAM_ID"})
    )
    game_opp_totals = game_opp_totals.merge(abbrev_to_id, on="OPPONENT_ABBREV", how="left")

    # Add game dates
    game_dates = player_logs[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    game_opp_totals = game_opp_totals.merge(game_dates, on="GAME_ID", how="left")

    # Step 2: For each opponent team, compute rolling mean of {stat} conceded
    game_opp_totals = game_opp_totals.sort_values(["OPP_TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    for w in windows:
        col_name = f"OPP_CONCEDED_{stat}_ROLL{w}"
        game_opp_totals[col_name] = (
            game_opp_totals.groupby("OPP_TEAM_ID")[f"OPP_CONCEDED_{stat}"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean())
        )

    # Step 3: Join back to player_logs
    merge_cols = ["GAME_ID", "OPPONENT_ABBREV"] + [f"OPP_CONCEDED_{stat}_ROLL{w}" for w in windows]
    out = out.merge(
        game_opp_totals[merge_cols].drop_duplicates("GAME_ID"),
        on=["GAME_ID", "OPPONENT_ABBREV"],
        how="left",
    )

    return out


def add_opponent_pace_and_rating(
    player_logs: pd.DataFrame,
    team_paired: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Join opponent's rolling PACE, DEF_RATING, and NET_RATING onto player logs.

    This allows the ML model to know whether the upcoming opponent plays fast
    (more possessions → more counting stat opportunities) or defends well.

    Parameters
    ----------
    player_logs : cleaned player logs
    team_paired : output of add_team_rolling_features(build_team_game_pairs(...))

    Returns
    -------
    player_logs with OPP_PACE_ROLL{W}, OPP_DEF_RATING_ROLL{W} columns added.
    """
    if windows is None:
        windows = settings.ROLLING_WINDOWS

    out = player_logs.copy()

    # Extract opponent team rolling stats from paired team logs
    opp_cols = ["GAME_ID", "TEAM_ID"]
    for w in windows:
        for metric in ["DEF_RATING", "NET_RATING", "PACE"]:
            col = f"ROLL{w}_{metric}"
            if col in team_paired.columns:
                opp_cols.append(col)

    # team_paired rows where TEAM_ID is the opponent
    # We match: player_log OPPONENT_ABBREV → team_paired TEAM_ABBREVIATION for same GAME_ID
    opp_team_ref = team_paired[opp_cols + ["TEAM_ABBREVIATION"]].drop_duplicates()

    # Rename for joining onto player_logs
    rename_map = {c: f"OPP_{c}" for c in opp_cols if c not in ("GAME_ID", "TEAM_ID")}
    rename_map["TEAM_ABBREVIATION"] = "OPPONENT_ABBREV"
    rename_map["TEAM_ID"] = "OPP_TEAM_ID"
    opp_team_ref = opp_team_ref.rename(columns=rename_map)

    out = out.merge(
        opp_team_ref,
        on=["GAME_ID", "OPPONENT_ABBREV"],
        how="left",
    )

    return out
