"""
data/loader.py
──────────────
Data access layer: loads raw Parquet files into Pandas DataFrames.

Design principles:
- All files are loaded once and cached in memory (module-level singletons).
- Callers receive the same DataFrame instance — do not mutate them directly;
  always call .copy() if you need to modify.
- Season filtering is supported via settings.seasons_filter.
- No SQL Server, no external network calls.
"""

from __future__ import annotations

import glob
import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Internal cache ────────────────────────────────────────────────────────────
_player_logs_cache: Optional[pd.DataFrame] = None
_team_logs_cache: Optional[pd.DataFrame] = None
_common_players_cache: Optional[pd.DataFrame] = None
_league_dash_players_cache: Optional[pd.DataFrame] = None
_league_dash_teams_cache: Optional[pd.DataFrame] = None
_advanced_boxscore_sample_cache: Optional[pd.DataFrame] = None
_team_roster_cache: Optional[pd.DataFrame] = None


def _load_parquet_dir(
    directory: Path,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load all Parquet files from *directory*, optionally filtering to specific
    seasons. Files must be named like `player_logs_2023-24.parquet`.
    """
    pattern = str(directory / "*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {directory}. "
            "Check NBA_DATA_ROOT in your .env file."
        )

    if seasons:
        # Filter files whose name contains one of the requested season strings
        files = [f for f in files if any(s in Path(f).name for s in seasons)]
        if not files:
            raise ValueError(
                f"No files matched seasons {seasons} in {directory}."
            )

    logger.info("Loading %d Parquet file(s) from %s …", len(files), directory)
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ── Public loaders ────────────────────────────────────────────────────────────

def get_player_logs(
    seasons: Optional[List[str]] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Return the full player game logs DataFrame.

    Parameters
    ----------
    seasons : list of season strings, e.g. ["2023-24", "2024-25"].
              Defaults to settings.seasons_filter (or all seasons if not set).
    force_reload : bypass the in-memory cache.
    """
    global _player_logs_cache

    effective_seasons = seasons if seasons is not None else settings.seasons_filter

    if _player_logs_cache is None or force_reload or effective_seasons:
        raw = _load_parquet_dir(settings.player_logs_dir, effective_seasons)

        # Normalise date column
        raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")

        # Derive home/away indicator from MATCHUP string
        raw["IS_HOME"] = raw["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)

        # Sort chronologically
        raw = raw.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        # Cache only when loading all seasons (no filter)
        if effective_seasons is None:
            _player_logs_cache = raw
        else:
            return raw

    return _player_logs_cache


def get_team_logs(
    seasons: Optional[List[str]] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Return the full team game logs DataFrame."""
    global _team_logs_cache

    effective_seasons = seasons if seasons is not None else settings.seasons_filter

    if _team_logs_cache is None or force_reload or effective_seasons:
        raw = _load_parquet_dir(settings.team_logs_dir, effective_seasons)

        raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
        raw["IS_HOME"] = raw["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
        raw = raw.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        if effective_seasons is None:
            _team_logs_cache = raw
        else:
            return raw

    return _team_logs_cache


def get_common_players() -> pd.DataFrame:
    """Return the player directory (commonallplayers.parquet)."""
    global _common_players_cache
    if _common_players_cache is None:
        path = settings.sample_endpoints_dir / "commonallplayers.parquet"
        _common_players_cache = pd.read_parquet(path)
        logger.info("Loaded common players: %d rows", len(_common_players_cache))
    return _common_players_cache


def get_league_dash_players() -> pd.DataFrame:
    """Return season-level player stats (leaguedashplayerstats.parquet)."""
    global _league_dash_players_cache
    if _league_dash_players_cache is None:
        path = settings.sample_endpoints_dir / "leaguedashplayerstats.parquet"
        _league_dash_players_cache = pd.read_parquet(path)
    return _league_dash_players_cache


def get_league_dash_teams() -> pd.DataFrame:
    """Return season-level team stats (leaguedashteamstats.parquet)."""
    global _league_dash_teams_cache
    if _league_dash_teams_cache is None:
        path = settings.sample_endpoints_dir / "leaguedashteamstats.parquet"
        _league_dash_teams_cache = pd.read_parquet(path)
    return _league_dash_teams_cache


def get_advanced_boxscore_sample() -> pd.DataFrame:
    """Return the advanced boxscore schema sample (not a full dataset)."""
    global _advanced_boxscore_sample_cache
    if _advanced_boxscore_sample_cache is None:
        path = settings.sample_endpoints_dir / "boxscoreadvancedv3.parquet"
        _advanced_boxscore_sample_cache = pd.read_parquet(path)
    return _advanced_boxscore_sample_cache


def get_team_roster_sample() -> pd.DataFrame:
    """Return the team roster sample (commonteamroster.parquet)."""
    global _team_roster_cache
    if _team_roster_cache is None:
        path = settings.sample_endpoints_dir / "commonteamroster.parquet"
        _team_roster_cache = pd.read_parquet(path)
    return _team_roster_cache


def _ascii_normalize(s: str) -> str:
    """Normalize a string to ASCII for accent-insensitive matching (e.g. 'Jokić' → 'jokic')."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def resolve_player_id(name: str) -> Optional[int]:
    """
    Fuzzy-match a player name string and return the PLAYER_ID.
    Supports accented characters (e.g. 'Jokic' → Nikola Jokić).
    Search order: exact → partial → ASCII-normalized → player_logs fallback.
    """
    players = get_common_players()
    name_lower = name.strip().lower()
    name_ascii = _ascii_normalize(name)

    # Normalise all player names to ASCII for comparison
    players = players.copy()
    players["_name_ascii"] = players["DISPLAY_FIRST_LAST"].apply(_ascii_normalize)

    # Exact match (ASCII-normalized)
    exact = players[players["_name_ascii"] == name_ascii]
    if not exact.empty:
        return int(exact.iloc[0]["PERSON_ID"])

    # Partial match (ASCII-normalized substring)
    partial = players[players["_name_ascii"].str.contains(name_ascii, na=False)]
    if not partial.empty:
        if len(partial) == 1:
            return int(partial.iloc[0]["PERSON_ID"])
        # Multiple matches — return best (shortest name = closest match)
        best = partial.iloc[partial["DISPLAY_FIRST_LAST"].str.len().argmin()]
        return int(best["PERSON_ID"])

    # Fallback: search PLAYER_NAME in player_logs (handles nicknames, last names)
    player_logs = get_player_logs()
    player_logs = player_logs.copy()
    player_logs["_name_ascii"] = player_logs["PLAYER_NAME"].apply(_ascii_normalize)
    pl_match = player_logs[player_logs["_name_ascii"].str.contains(name_ascii, na=False)]
    if not pl_match.empty:
        # Pick the player with the most games (most prominent match)
        top = pl_match.groupby("PLAYER_ID")["GAME_ID"].count().idxmax()
        return int(top)

    return None


def resolve_team_id(name_or_abbrev: str) -> Optional[int]:
    """
    Resolve a team name or abbreviation to TEAM_ID from team logs.
    """
    team_logs = get_team_logs()
    val = name_or_abbrev.strip().upper()

    # Try abbreviation first
    abbrev_match = team_logs[team_logs["TEAM_ABBREVIATION"].str.upper() == val]
    if not abbrev_match.empty:
        return int(abbrev_match.iloc[0]["TEAM_ID"])

    # Try full name (case-insensitive contains)
    name_match = team_logs[team_logs["TEAM_NAME"].str.lower().str.contains(val.lower(), na=False)]
    if not name_match.empty:
        return int(name_match.iloc[0]["TEAM_ID"])

    return None
