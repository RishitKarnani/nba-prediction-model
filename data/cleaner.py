"""
data/cleaner.py
───────────────
Data processing layer: cleans raw DataFrames and adds reliable derived columns.

Rules:
- Never modifies the cached DataFrames from loader.py in-place.
- Always returns a NEW DataFrame (copy).
- Handles NaN shooting percentages (0/0 cases) predictably.
- Derives calculated metrics from raw box score columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Player log cleaning ───────────────────────────────────────────────────────

def clean_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich raw player game logs.

    Additions:
    - FG_PCT, FG3_PCT, FT_PCT: NaN → 0.0 (player had no attempts)
    - IS_HOME: derived from MATCHUP if not already present
    - TRUE_SHOOTING_PCT: (PTS / (2 * (FGA + 0.44 * FTA))) — NaN → 0
    - EFG_PCT: (FGM + 0.5 * FG3M) / FGA — NaN → 0
    - OPPONENT_ABBREV: parsed from MATCHUP string
    - DNP: flag for Did Not Play (MIN == 0)
    """
    out = df.copy()

    # Shooting percentages: 0/0 → 0
    out["FG_PCT"] = out["FG_PCT"].fillna(0.0)
    out["FG3_PCT"] = out["FG3_PCT"].fillna(0.0)
    out["FT_PCT"] = out["FT_PCT"].fillna(0.0)

    # Home/away (may already be present from loader)
    if "IS_HOME" not in out.columns:
        out["IS_HOME"] = out["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)

    # True Shooting %
    ts_denom = 2.0 * (out["FGA"] + 0.44 * out["FTA"])
    out["TRUE_SHOOTING_PCT"] = np.where(ts_denom > 0, out["PTS"] / ts_denom, 0.0)

    # Effective Field Goal %
    out["EFG_PCT"] = np.where(out["FGA"] > 0, (out["FGM"] + 0.5 * out["FG3M"]) / out["FGA"], 0.0)

    # Opponent abbreviation parsed from MATCHUP ("DEN vs. LAL" → "LAL", "DEN @ GSW" → "GSW")
    out["OPPONENT_ABBREV"] = out.apply(_parse_opponent_abbrev, axis=1)

    # Did Not Play flag
    out["DNP"] = (out["MIN"] == 0).astype(int)

    return out


def _parse_opponent_abbrev(row: pd.Series) -> str:
    """Extract opponent team abbreviation from MATCHUP string."""
    matchup = str(row.get("MATCHUP", ""))
    parts = matchup.replace("vs.", "@").split("@")
    if len(parts) == 2:
        own_abbrev = str(row.get("TEAM_ABBREVIATION", "")).strip()
        for part in parts:
            abbrev = part.strip()
            if abbrev and abbrev != own_abbrev:
                return abbrev
    return ""


# ── Team log cleaning ─────────────────────────────────────────────────────────

def clean_team_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich raw team game logs.

    Additions:
    - FG3_PCT, FT_PCT: NaN → 0.0
    - IS_HOME: derived if not present
    - POSSESSIONS: estimated via Dean Oliver formula
    - OFF_RATING: 100 * PTS / POSSESSIONS
    - EFG_PCT, TOV_PCT, FT_RATE: Dean Oliver Four Factors
    - OPPONENT_ABBREV: from MATCHUP
    """
    out = df.copy()

    # Shooting percentages
    for col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    # Home/away
    if "IS_HOME" not in out.columns:
        out["IS_HOME"] = out["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)

    # Dean Oliver Possessions estimate
    out["POSSESSIONS"] = (
        out["FGA"]
        + 0.44 * out["FTA"]
        - out["OREB"]
        + out["TOV"]
    ).clip(lower=1)

    # Offensive Rating (points per 100 possessions)
    out["OFF_RATING"] = (out["PTS"] / out["POSSESSIONS"]) * 100.0

    # eFG%
    out["EFG_PCT"] = np.where(out["FGA"] > 0, (out["FGM"] + 0.5 * out["FG3M"]) / out["FGA"], 0.0)

    # Turnover %
    tov_denom = out["FGA"] + 0.44 * out["FTA"] + out["TOV"]
    out["TOV_PCT"] = np.where(tov_denom > 0, out["TOV"] / tov_denom, 0.0)

    # Offensive Rebound %  (we only have own OREB here; will be updated with opponent DREB in joiner)
    out["FT_RATE"] = np.where(out["FGA"] > 0, out["FTM"] / out["FGA"], 0.0)

    # Opponent abbreviation
    out["OPPONENT_ABBREV"] = out.apply(_parse_team_opponent_abbrev, axis=1)

    return out


def _parse_team_opponent_abbrev(row: pd.Series) -> str:
    matchup = str(row.get("MATCHUP", ""))
    parts = matchup.replace("vs.", "@").split("@")
    if len(parts) == 2:
        own = str(row.get("TEAM_ABBREVIATION", "")).strip()
        for part in parts:
            abbrev = part.strip()
            if abbrev and abbrev != own:
                return abbrev
    return ""
