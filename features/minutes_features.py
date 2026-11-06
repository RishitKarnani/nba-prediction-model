"""
features/minutes_features.py
────────────────────────────
Computes pre-game expected minutes features for individual players.

All features are strictly computed with shift(1) — the target game's actual
minutes are NEVER used as an input (zero data leakage).

Features generated:
1. Rolling Average Minutes:
   - ROLL5_MIN_MEAN   : Minutes average over last 5 active games
   - ROLL10_MIN_MEAN  : Minutes average over last 10 active games
   - ROLL20_MIN_MEAN  : Minutes average over last 20 active games
2. Minutes Consistency:
   - ROLL5_MIN_STD    : Standard deviation over last 5 active games (volatility)
   - ROLL10_MIN_STD   : Standard deviation over last 10 active games
   - ROLL20_MIN_STD   : Standard deviation over last 20 active games
3. Minutes Trend:
   - MIN_TREND_5_10   : ROLL5_MIN_MEAN - ROLL10_MIN_MEAN (short vs medium momentum)
   - MIN_TREND_10_20  : ROLL10_MIN_MEAN - ROLL20_MIN_MEAN (medium vs long momentum)
   - MIN_TREND_5_20   : ROLL5_MIN_MEAN - ROLL20_MIN_MEAN (short vs baseline momentum)
4. Season Context:
   - SEASON_MIN_MEAN       : Season average minutes before the target game
   - SEASON_GAMES_PLAYED   : Prior games played in current season before target game
   - SEASON_STARTER_PCT    : % of prior season games started (or core rotation >= 24 min proxy)
5. Recent Role:
   - ROLL5_STARTER_PCT     : % of last 5 games started (or core rotation >= 24 min proxy)
   - ROLL10_STARTER_PCT    : % of last 10 games started (or core rotation >= 24 min proxy)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import settings

# Threshold to designate a player as a core starter / primary rotation piece
# Standard NBA starters average between 24 and 38 minutes per game.
DEFAULT_STARTER_MIN_THRESHOLD = 24.0


def add_expected_minutes_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    starter_min_threshold: float = DEFAULT_STARTER_MIN_THRESHOLD,
    min_active_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate pre-game expected minutes features for every player in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned player game logs (must have PLAYER_ID, GAME_DATE, MIN, SEASON).
    windows : list of int, optional
        Rolling window sizes. Defaults to settings.ROLLING_WINDOWS ([5, 10, 20]).
    starter_min_threshold : float, default 24.0
        Minutes threshold to treat a game as a starter / core rotation role
        when official START_POSITION is not present.
    min_active_minutes : int, optional
        Minimum minutes to count as an active game. Defaults to settings.MIN_MINUTES_THRESHOLD (5).

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with all minutes features appended.
    """
    if windows is None:
        windows = settings.ROLLING_WINDOWS
    if min_active_minutes is None:
        min_active_minutes = settings.MIN_MINUTES_THRESHOLD

    out = df.copy()

    # Ensure chronological order per player
    out = out.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # 1. Mask non-active (DNP / garbage time) games so they don't contaminate averages
    active_mask = out["MIN"] >= min_active_minutes
    active_minutes = np.where(active_mask, out["MIN"].astype(float), np.nan)
    out["_active_min"] = active_minutes

    # Starter indicator: use START_POSITION / IS_STARTER if present, else proxy by minutes
    if "START_POSITION" in out.columns:
        is_starter = (
            out["START_POSITION"].notna()
            & (out["START_POSITION"].astype(str).str.strip() != "")
        ).astype(float)
    elif "IS_STARTER" in out.columns:
        is_starter = out["IS_STARTER"].astype(float)
    else:
        # Starter proxy: games where player logged >= starter_min_threshold minutes
        is_starter = (out["MIN"] >= starter_min_threshold).astype(float)

    # Mask starter flag for DNP games
    out["_is_starter_flag"] = np.where(active_mask, is_starter, np.nan)

    new_cols = {}

    # 2. Rolling Average Minutes & Consistency (std dev)
    grp_active = out.groupby("PLAYER_ID")["_active_min"]
    grp_starter = out.groupby("PLAYER_ID")["_is_starter_flag"]

    for w in windows:
        # shift(1) excludes the current game from the window
        mean_s = grp_active.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        std_s = grp_active.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(2, w // 2)).std()
        )
        starter_pct_s = grp_starter.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )

        new_cols[f"ROLL{w}_MIN_MEAN"] = mean_s
        new_cols[f"ROLL{w}_MIN_STD"] = std_s
        new_cols[f"ROLL{w}_STARTER_PCT"] = starter_pct_s

    # 3. Minutes Trends (Short vs Medium vs Long-term momentum)
    r5 = new_cols.get("ROLL5_MIN_MEAN")
    r10 = new_cols.get("ROLL10_MIN_MEAN")
    r20 = new_cols.get("ROLL20_MIN_MEAN")

    if r5 is not None and r10 is not None:
        new_cols["MIN_TREND_5_10"] = r5 - r10
    if r10 is not None and r20 is not None:
        new_cols["MIN_TREND_10_20"] = r10 - r20
    if r5 is not None and r20 is not None:
        new_cols["MIN_TREND_5_20"] = r5 - r20

    # 4. Season Context (prior games in same season only)
    # Expanding season average minutes
    season_grp_active = out.groupby(["PLAYER_ID", "SEASON"])["_active_min"]
    new_cols["SEASON_MIN_MEAN"] = season_grp_active.transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Number of prior games played in current season
    season_grp_games = out.groupby(["PLAYER_ID", "SEASON"])["GAME_ID"]
    new_cols["SEASON_GAMES_PLAYED"] = season_grp_games.transform(
        lambda x: x.shift(1).expanding().count()
    ).fillna(0).astype(int)

    # Percentage of prior season games started (or core rotation proxy)
    season_grp_starter = out.groupby(["PLAYER_ID", "SEASON"])["_is_starter_flag"]
    new_cols["SEASON_STARTER_PCT"] = season_grp_starter.transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Avoid duplicate column names if already computed
    existing_overlap = [c for c in new_cols.keys() if c in out.columns]
    if existing_overlap:
        out = out.drop(columns=existing_overlap)

    # Attach all new columns in a single concat to prevent fragmentation
    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    # Drop temporary scratch columns
    out = out.drop(columns=["_active_min", "_is_starter_flag"], errors="ignore")

    return out


def get_player_minutes_snapshot(
    df: pd.DataFrame,
    player_id: int,
    starter_min_threshold: float = DEFAULT_STARTER_MIN_THRESHOLD,
) -> dict:
    """
    Return a pre-game snapshot of expected minutes features for a single player
    based on their most recent historical data. Used at prediction time.
    """
    player_games = df[df["PLAYER_ID"] == player_id].sort_values("GAME_DATE")
    if player_games.empty:
        return {}

    enriched = add_expected_minutes_features(
        player_games,
        starter_min_threshold=starter_min_threshold,
    )
    latest = enriched.iloc[-1]

    return {
        "PLAYER_ID": player_id,
        "PLAYER_NAME": latest.get("PLAYER_NAME", ""),
        "ROLL5_MIN_MEAN": latest.get("ROLL5_MIN_MEAN"),
        "ROLL10_MIN_MEAN": latest.get("ROLL10_MIN_MEAN"),
        "ROLL20_MIN_MEAN": latest.get("ROLL20_MIN_MEAN"),
        "ROLL5_MIN_STD": latest.get("ROLL5_MIN_STD"),
        "ROLL10_MIN_STD": latest.get("ROLL10_MIN_STD"),
        "MIN_TREND_5_10": latest.get("MIN_TREND_5_10"),
        "MIN_TREND_10_20": latest.get("MIN_TREND_10_20"),
        "SEASON_MIN_MEAN": latest.get("SEASON_MIN_MEAN"),
        "SEASON_GAMES_PLAYED": latest.get("SEASON_GAMES_PLAYED"),
        "SEASON_STARTER_PCT": latest.get("SEASON_STARTER_PCT"),
        "ROLL5_STARTER_PCT": latest.get("ROLL5_STARTER_PCT"),
        "ROLL10_STARTER_PCT": latest.get("ROLL10_STARTER_PCT"),
    }
