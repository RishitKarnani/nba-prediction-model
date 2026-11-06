"""
features/player_features.py
────────────────────────────
Computes player-level rolling and contextual features from cleaned player logs.

Key design decisions:
  - All rolling features use shift(1) — the target game is EXCLUDED.
  - DNP (0-minute) games are excluded from rolling stat calculations.
  - Volatility (std dev) is computed alongside means.
  - Season averages are computed using all prior games in the same season.

Output features per stat S and window W:
  ROLL{W}_{S}_MEAN   : rolling mean (last W active games)
  ROLL{W}_{S}_STD    : rolling std  (consistency measure)
  ROLL{W}_{S}_MAX    : rolling max  (ceiling)
  ROLL{W}_{S}_MIN    : rolling min  (floor)
  SEASON_{S}_MEAN    : season-to-date mean (excluding current game)
  CAREER_{S}_MEAN    : career mean (all prior seasons)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import settings


# Stats we compute rolling features for
PROP_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGM", "FTM", "MIN", "OREB", "DREB"]


def add_player_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    stats: Optional[List[str]] = None,
    min_minutes: int = None,
) -> pd.DataFrame:
    """
    Add rolling player performance features to a player game log DataFrame.

    Parameters
    ----------
    df          : cleaned player game logs (must have PLAYER_ID, GAME_DATE, MIN, etc.)
    windows     : rolling window sizes. Defaults to settings.ROLLING_WINDOWS.
    stats       : list of stat columns. Defaults to PROP_STATS.
    min_minutes : minimum minutes for a game to count. Defaults to settings.MIN_MINUTES_THRESHOLD.

    Returns
    -------
    DataFrame with rolling feature columns added.
    """
    if windows is None:
        windows = settings.ROLLING_WINDOWS
    if stats is None:
        stats = PROP_STATS
    if min_minutes is None:
        min_minutes = settings.MIN_MINUTES_THRESHOLD

    out = df.copy()

    # Sort chronologically per player
    out = out.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Mask DNP rows — set stats to NaN for rolling (do not contaminate averages)
    for stat in stats:
        if stat not in out.columns:
            continue
        out[f"_active_{stat}"] = np.where(out["MIN"] >= min_minutes, out[stat].astype(float), np.nan)

    # Rolling features per window — batch all columns via pd.concat to avoid fragmentation
    new_cols = {}
    for w in windows:
        for stat in stats:
            if f"_active_{stat}" not in out.columns:
                continue
            active_col = f"_active_{stat}"

            # shift(1) excludes the current game from the window
            grp = out.groupby("PLAYER_ID")[active_col]

            new_cols[f"ROLL{w}_{stat}_MEAN"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
            )
            new_cols[f"ROLL{w}_{stat}_STD"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(2, w // 2)).std()
            )
            new_cols[f"ROLL{w}_{stat}_MAX"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).max()
            )
            new_cols[f"ROLL{w}_{stat}_MIN"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).min()
            )

    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    # Clean up temp columns
    out = out.drop(columns=[f"_active_{stat}" for stat in stats if f"_active_{stat}" in out.columns])

    return out


def add_season_averages(df: pd.DataFrame, stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add season-to-date (expanding) averages per player, excluding current game.

    Columns: SEASON_{STAT}_MEAN
    """
    if stats is None:
        stats = PROP_STATS

    out = df.copy()
    out = out.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)

    for stat in stats:
        if stat not in out.columns:
            continue
        active = np.where(out["MIN"] >= settings.MIN_MINUTES_THRESHOLD, out[stat].astype(float), np.nan)
        out["_active"] = active
        out[f"SEASON_{stat}_MEAN"] = (
            out.groupby(["PLAYER_ID", "SEASON"])["_active"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

    out = out.drop(columns=["_active"], errors="ignore")
    return out


def add_career_averages(df: pd.DataFrame, stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add career-to-date (expanding across all seasons) averages per player.

    Columns: CAREER_{STAT}_MEAN
    """
    if stats is None:
        stats = PROP_STATS

    out = df.copy()
    out = out.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    for stat in stats:
        if stat not in out.columns:
            continue
        active = np.where(out["MIN"] >= settings.MIN_MINUTES_THRESHOLD, out[stat].astype(float), np.nan)
        out["_active"] = active
        out[f"CAREER_{stat}_MEAN"] = (
            out.groupby("PLAYER_ID")["_active"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

    out = out.drop(columns=["_active"], errors="ignore")
    return out


def add_home_away_splits(df: pd.DataFrame, stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add trailing 10-game rolling mean split by home/away location.

    Columns: HOME_ROLL10_{STAT}_MEAN, AWAY_ROLL10_{STAT}_MEAN
    """
    if stats is None:
        stats = ["PTS", "REB", "AST"]

    out = df.copy()
    out = out.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    for location, flag in [("HOME", 1), ("AWAY", 0)]:
        mask = (out["IS_HOME"] == flag) & (out["MIN"] >= settings.MIN_MINUTES_THRESHOLD)
        for stat in stats:
            if stat not in out.columns:
                continue
            col_name = f"{location}_ROLL10_{stat}_MEAN"
            sub = out.loc[mask, ["PLAYER_ID", "GAME_DATE", stat]].copy()
            sub[col_name] = (
                sub.groupby("PLAYER_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            )
            out = out.merge(
                sub[["PLAYER_ID", "GAME_DATE", col_name]],
                on=["PLAYER_ID", "GAME_DATE"],
                how="left"
            )

    return out


def get_player_game_history(
    df: pd.DataFrame,
    player_id: int,
    stat: str,
    n_games: int = 20,
    min_minutes: int = None,
) -> pd.Series:
    """
    Return the last N active game values for a specific player and stat.
    Used for probability distribution estimation.

    Parameters
    ----------
    df        : cleaned player logs
    player_id : NBA PLAYER_ID
    stat      : e.g. 'PTS', 'REB', 'AST'
    n_games   : number of most recent active games to return
    min_minutes : minimum minutes to count a game

    Returns
    -------
    pd.Series of stat values, most recent last. May have fewer than n_games rows.
    """
    if min_minutes is None:
        min_minutes = settings.MIN_MINUTES_THRESHOLD

    player_df = df[df["PLAYER_ID"] == player_id].copy()
    player_df = player_df[player_df["MIN"] >= min_minutes]
    player_df = player_df.sort_values("GAME_DATE")

    if stat not in player_df.columns:
        return pd.Series(dtype=float)

    return player_df[stat].astype(float).tail(n_games).reset_index(drop=True)


def compute_recent_form_delta(
    df: pd.DataFrame,
    player_id: int,
    stat: str,
    short_window: int = 5,
    long_window: int = 20,
) -> Dict[str, float]:
    """
    Compare short-term vs. long-term form for a player on a given stat.

    Returns dict with keys:
      short_avg, long_avg, delta, delta_pct
    """
    history = get_player_game_history(df, player_id, stat, n_games=long_window)

    if len(history) == 0:
        return {"short_avg": np.nan, "long_avg": np.nan, "delta": np.nan, "delta_pct": np.nan}

    long_avg = history.mean()
    short_avg = history.tail(short_window).mean() if len(history) >= short_window else history.mean()
    delta = short_avg - long_avg
    delta_pct = (delta / long_avg * 100) if long_avg != 0 else np.nan

    return {
        "short_avg": round(short_avg, 2),
        "long_avg": round(long_avg, 2),
        "delta": round(delta, 2),
        "delta_pct": round(delta_pct, 1) if not np.isnan(delta_pct) else np.nan,
    }
