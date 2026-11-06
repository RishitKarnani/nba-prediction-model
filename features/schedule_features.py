"""
features/schedule_features.py
──────────────────────────────
Computes schedule-based features for both teams and players:
  - Rest days between games
  - Back-to-back (B2B) indicator
  - 3-in-4-nights indicator
  - Rest advantage differential (home vs. away)

CRITICAL: All features use shift(1) so the current game is NEVER in the feature.
"""

from __future__ import annotations

import pandas as pd


def add_rest_features(df: pd.DataFrame, id_col: str = "TEAM_ID") -> pd.DataFrame:
    """
    Add rest-day features to a game log DataFrame (team or player).

    Parameters
    ----------
    df : must have columns [id_col, 'GAME_DATE'] sorted by (id_col, GAME_DATE).
    id_col : the entity ID column — 'TEAM_ID' or 'PLAYER_ID'.

    Returns
    -------
    DataFrame with new columns:
      REST_DAYS        : calendar days since last game (NaN for first game)
      IS_B2B           : 1 if REST_DAYS == 1 else 0
      IS_3IN4          : 1 if this is the team's 3rd game in 4 calendar days
    """
    out = df.copy()
    out = out.sort_values([id_col, "GAME_DATE"]).reset_index(drop=True)

    # Days since last game
    out["REST_DAYS"] = (
        out.groupby(id_col)["GAME_DATE"]
        .diff()
        .dt.days
    )

    # Back-to-back
    out["IS_B2B"] = (out["REST_DAYS"] == 1).astype(int)

    # 3-in-4 nights: check if team played 2 days ago (2 days prior in rolling 3-game window)
    # Flag game N if game N-2 was within 3 calendar days of game N
    def _is_3in4(series: pd.Series) -> pd.Series:
        dates = series.sort_values().reset_index(drop=True)
        flags = pd.Series(0, index=dates.index)
        for i in range(2, len(dates)):
            span = (dates.iloc[i] - dates.iloc[i - 2]).days
            if span <= 3:
                flags.iloc[i] = 1
        return flags

    out["IS_3IN4"] = (
        out.groupby(id_col)["GAME_DATE"]
        .transform(_is_3in4)
    )

    return out


def add_team_rest_differential(paired_team_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Add REST_ADV column to the paired team game logs (output of joiner.build_team_game_pairs).

    REST_ADV = own REST_DAYS - opponent REST_DAYS (positive favours this team)
    """
    out = paired_team_logs.copy()

    if "REST_DAYS" not in out.columns:
        raise ValueError("Call add_rest_features() on team_logs before add_team_rest_differential().")

    # Opponent rest days: join opponent's REST_DAYS via GAME_ID and OPP_TEAM_ID
    opp_rest = (
        out[["GAME_ID", "TEAM_ID", "REST_DAYS"]]
        .rename(columns={"TEAM_ID": "OPP_TEAM_ID", "REST_DAYS": "OPP_REST_DAYS"})
    )
    out = out.merge(opp_rest, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
    out["REST_ADV"] = out["REST_DAYS"] - out["OPP_REST_DAYS"]

    return out
