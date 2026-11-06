"""
features/team_features.py
──────────────────────────
Computes rolling team-level features from paired team game logs.

All rolling windows are computed with shift(1) — the current game's stats
are NEVER included in the feature vector (no data leakage).

Output columns per rolling window W (e.g. W=5, 10, 20):
  - ROLL{W}_OFF_RATING  : rolling offensive rating
  - ROLL{W}_DEF_RATING  : rolling defensive rating
  - ROLL{W}_NET_RATING  : rolling net rating
  - ROLL{W}_EFG_PCT     : rolling effective FG%
  - ROLL{W}_TOV_PCT     : rolling turnover %
  - ROLL{W}_FT_RATE     : rolling free throw rate
  - ROLL{W}_PTS         : rolling points scored
  - ROLL{W}_OREB_PCT    : rolling offensive rebound %
  - ROLL{W}_PACE        : rolling estimated pace
  - ROLL{W}_WIN_PCT     : rolling win rate
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from config.settings import settings


TEAM_ROLL_COLS = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "EFG_PCT",
    "TOV_PCT",
    "FT_RATE",
    "PTS",
    "OREB_PCT",
    "WIN",
]


def add_team_rolling_features(
    df: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean features for each team, sorted chronologically.

    Parameters
    ----------
    df      : output of joiner.build_team_game_pairs() — must have WIN, OFF_RATING etc.
    windows : list of rolling window sizes. Defaults to settings.ROLLING_WINDOWS.

    Returns
    -------
    DataFrame with ROLL{W}_{STAT} columns added.
    """
    if windows is None:
        windows = settings.ROLLING_WINDOWS

    out = df.copy()

    # Ensure chronological sort per team
    out = out.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Add pace proxy (possessions per 48 min equivalent)
    if "POSSESSIONS" in out.columns and "MIN" in out.columns:
        out["PACE"] = np.where(
            out["MIN"] > 0,
            (out["POSSESSIONS"] / out["MIN"]) * 48.0,
            np.nan,
        )
    else:
        out["PACE"] = np.nan

    roll_cols = TEAM_ROLL_COLS + (["PACE"] if "PACE" in out.columns else [])

    for w in windows:
        # shift(1) ensures the current game is excluded from the rolling window
        rolled = (
            out.groupby("TEAM_ID")[roll_cols]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=max(1, w // 2)).mean())
        )
        rolled.columns = [f"ROLL{w}_{c}" for c in roll_cols]
        out = pd.concat([out, rolled], axis=1)

    # Rolling win % specifically
    for w in windows:
        out[f"ROLL{w}_WIN_PCT"] = out[f"ROLL{w}_WIN"]

    return out


def add_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team home and away rolling averages separately.
    Adds columns prefixed with HOME_ and AWAY_ for key metrics.

    Uses trailing 10-game window for home games and 10-game window for away games.
    """
    out = df.copy()
    out = out.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    for location, flag in [("HOME", 1), ("AWAY", 0)]:
        mask = out["IS_HOME"] == flag
        for stat in ["PTS", "OFF_RATING", "DEF_RATING", "NET_RATING", "WIN"]:
            if stat not in out.columns:
                continue
            col_name = f"{location}_ROLL10_{stat}"
            # Compute rolling only within the subset, then re-align to full df
            sub = out.loc[mask, ["TEAM_ID", "GAME_DATE", stat]].copy()
            sub[col_name] = (
                sub.groupby("TEAM_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            )
            out = out.merge(sub[["TEAM_ID", "GAME_DATE", col_name]], on=["TEAM_ID", "GAME_DATE"], how="left")

    return out
