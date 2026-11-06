"""
features/pipeline.py
─────────────────────
Unified feature engineering pipeline.

This module assembles all feature layers into one clean call:
  build_player_feature_matrix(stat)  →  DataFrame ready for ML training
  build_team_feature_matrix()        →  DataFrame ready for win-prob model

Usage:
  from features.pipeline import build_player_feature_matrix
  df = build_player_feature_matrix("PTS")
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from config.settings import settings
from data.cleaner import clean_player_logs, clean_team_logs
from data.joiner import build_team_game_pairs
from data.loader import get_player_logs, get_team_logs
from features.minutes_features import add_expected_minutes_features
from features.opponent_features import add_opponent_pace_and_rating, build_opponent_defensive_features
from features.player_features import (
    PROP_STATS,
    add_career_averages,
    add_home_away_splits,
    add_player_rolling_features,
    add_season_averages,
)
from features.schedule_features import add_rest_features, add_team_rest_differential
from features.team_features import add_team_rolling_features

logger = logging.getLogger(__name__)


def _build_team_paired_with_features(seasons: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Internal helper: builds the full team feature matrix used as a dependency
    by both the team win model and the player feature matrix.
    """
    raw_teams = get_team_logs(seasons=seasons)
    clean_teams = clean_team_logs(raw_teams)

    # Add rest features
    clean_teams = add_rest_features(clean_teams, id_col="TEAM_ID")

    # Pair home/away
    paired = build_team_game_pairs(clean_teams)

    # Add rest differential
    paired = add_team_rest_differential(paired)

    # Add rolling team features
    paired = add_team_rolling_features(paired)

    return paired


def build_team_feature_matrix(seasons: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build the complete team win-probability feature matrix.

    Returns a DataFrame with one row per team per game, containing:
    - All rolling offensive/defensive features
    - Rest days and back-to-back flags
    - Home/away indicator
    - WIN label column
    """
    logger.info("Building team feature matrix …")
    paired = _build_team_paired_with_features(seasons)
    logger.info("Team feature matrix ready: %d rows × %d cols", *paired.shape)
    return paired


def build_player_feature_matrix(
    stat: str,
    seasons: Optional[List[str]] = None,
    min_games: int = None,
) -> pd.DataFrame:
    """
    Build the complete player prop feature matrix for a given stat.

    Steps:
    1. Load and clean player logs
    2. Add player rolling features (last 5/10/20 games)
    3. Add season and career averages
    4. Add home/away splits
    5. Add rest features
    6. Build team paired matrix for opponent context
    7. Add opponent defensive context (how much of this stat has opponent conceded?)
    8. Add opponent pace and DEF_RATING
    9. Drop DNP games (MIN < threshold) from final training set
    10. Drop rows where target stat is NaN

    Parameters
    ----------
    stat     : target stat column name, e.g. 'PTS', 'REB', 'AST'
    seasons  : optional season filter list
    min_games: minimum games a player must have to be included

    Returns
    -------
    DataFrame ready for ML training with `stat` as the target column.
    """
    if min_games is None:
        min_games = settings.MIN_GAMES_PLAYER

    logger.info("Building player feature matrix for stat='%s' …", stat)

    # ── Step 1: Load and clean player logs ───────────────────────────────────
    raw_players = get_player_logs(seasons=seasons)
    players = clean_player_logs(raw_players)

    # ── Step 2: Player rolling features ──────────────────────────────────────
    players = add_player_rolling_features(players)

    # ── Step 3: Expected Minutes features (consistency, trends, season context)
    players = add_expected_minutes_features(players)

    # ── Step 4: Season and career averages ───────────────────────────────────
    players = add_season_averages(players, stats=PROP_STATS)
    players = add_career_averages(players, stats=PROP_STATS)

    # ── Step 5: Home/away splits ─────────────────────────────────────────────
    players = add_home_away_splits(players, stats=["PTS", "REB", "AST"])

    # ── Step 6: Rest features for players ────────────────────────────────────
    players = add_rest_features(players, id_col="PLAYER_ID")

    # ── Step 7: Team paired matrix ───────────────────────────────────────────
    team_paired = _build_team_paired_with_features(seasons)

    # ── Step 8: Opponent defensive context ───────────────────────────────────
    players = build_opponent_defensive_features(players, team_paired, stat=stat)

    # ── Step 9: Opponent pace and rating ─────────────────────────────────────
    players = add_opponent_pace_and_rating(players, team_paired)

    # ── Step 10: Filter DNP rows ─────────────────────────────────────────────
    players = players[players["MIN"] >= settings.MIN_MINUTES_THRESHOLD].copy()

    # ── Step 11: Filter out rows where target is missing ─────────────────────
    players = players.dropna(subset=[stat])

    # ── Step 12: Filter players with insufficient history ────────────────────
    game_counts = players.groupby("PLAYER_ID")["GAME_ID"].count()
    valid_players = game_counts[game_counts >= min_games].index
    players = players[players["PLAYER_ID"].isin(valid_players)].copy()

    logger.info(
        "Player feature matrix for '%s': %d rows × %d cols, %d unique players",
        stat, len(players), len(players.columns), players["PLAYER_ID"].nunique()
    )

    return players.reset_index(drop=True)


def get_player_context_snapshot(
    player_id: int,
    stat: str,
    opponent_team_abbrev: Optional[str] = None,
    is_home: int = 1,
    seasons: Optional[List[str]] = None,
) -> dict:
    """
    Build a single-row feature snapshot for a player's NEXT predicted game.
    Used by the prediction engine at inference time.

    Returns a dict of feature values (not a DataFrame row).
    """
    from features.minutes_features import get_player_minutes_snapshot
    from features.player_features import compute_recent_form_delta, get_player_game_history

    raw_players = get_player_logs(seasons=seasons)
    players = clean_player_logs(raw_players)

    player_df = players[players["PLAYER_ID"] == player_id].copy()
    if player_df.empty:
        return {}

    # Most recent game
    latest = player_df.sort_values("GAME_DATE").iloc[-1]

    history = get_player_game_history(players, player_id, stat, n_games=20)
    history_5 = history.tail(5)
    history_10 = history.tail(10)

    form = compute_recent_form_delta(players, player_id, stat)
    min_snapshot = get_player_minutes_snapshot(players, player_id)

    snapshot = {
        "PLAYER_ID": player_id,
        "PLAYER_NAME": latest.get("PLAYER_NAME", ""),
        "TEAM_ABBREVIATION": latest.get("TEAM_ABBREVIATION", ""),
        "STAT": stat,
        "IS_HOME": is_home,
        "OPPONENT_ABBREV": opponent_team_abbrev or "",
        f"ROLL5_{stat}_MEAN": history_5.mean() if len(history_5) > 0 else None,
        f"ROLL10_{stat}_MEAN": history_10.mean() if len(history_10) > 0 else None,
        f"ROLL20_{stat}_MEAN": history.mean() if len(history) > 0 else None,
        f"ROLL5_{stat}_STD": history_5.std() if len(history_5) > 1 else None,
        f"ROLL20_{stat}_STD": history.std() if len(history_10) > 1 else None,
        "SEASON_MEAN": form.get("long_avg"),
        "RECENT_MEAN": form.get("short_avg"),
        "FORM_DELTA": form.get("delta"),
        "FORM_DELTA_PCT": form.get("delta_pct"),
        "HISTORY_VALUES": history.tolist(),
        "EXPECTED_MIN_ROLL5": min_snapshot.get("ROLL5_MIN_MEAN"),
        "EXPECTED_MIN_ROLL10": min_snapshot.get("ROLL10_MIN_MEAN"),
        "EXPECTED_MIN_ROLL20": min_snapshot.get("ROLL20_MIN_MEAN"),
        "MIN_TREND_5_10": min_snapshot.get("MIN_TREND_5_10"),
        "MIN_TREND_10_20": min_snapshot.get("MIN_TREND_10_20"),
        "SEASON_GAMES_PLAYED": min_snapshot.get("SEASON_GAMES_PLAYED"),
        "SEASON_STARTER_PCT": min_snapshot.get("SEASON_STARTER_PCT"),
    }

    return snapshot
