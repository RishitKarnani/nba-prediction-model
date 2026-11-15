"""
data/schedule.py
────────────────
Fetches official NBA schedules for today, tomorrow, or any target date,
and extracts active team rotation rosters for upcoming game predictions.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

from data.loader import get_player_logs

logger = logging.getLogger(__name__)

# In-memory daily schedule cache: {date_str: (timestamp, games_list)}
_SCHEDULE_CACHE: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _normalize_date_str(date_val: Optional[str | datetime.date]) -> str:
    """Normalize any date input (including DD/MM/YY or DD/MM/YYYY) to 'YYYY-MM-DD'."""
    if date_val is None:
        return datetime.date.today().strftime("%Y-%m-%d")
    if isinstance(date_val, datetime.date):
        return date_val.strftime("%Y-%m-%d")

    s = str(date_val).strip()
    try:
        # Supports DD/MM/YY, DD/MM/YYYY, YYYY-MM-DD, etc.
        dt = pd.to_datetime(s, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # Fallback to direct string
        return s


def get_daily_schedule(game_date: Optional[str | datetime.date] = None, force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch the official NBA schedule for a specific date using nba_api.
    Supports flexible date parsing (e.g. '16/11/26', '2026-11-16', datetime.date).
    """
    target_str = _normalize_date_str(game_date)
    now_ts = datetime.datetime.now().timestamp()

    # Check in-memory cache
    if not force_reload and target_str in _SCHEDULE_CACHE:
        cached_ts, cached_games = _SCHEDULE_CACHE[target_str]
        if now_ts - cached_ts < CACHE_TTL_SECONDS:
            return cached_games

    try:
        sb = scoreboardv2.ScoreboardV2(game_date=target_str)
        gh = sb.game_header.get_data_frame()
        ls = sb.line_score.get_data_frame()
    except Exception as exc:
        logger.warning("Failed to fetch live Scoreboard for %s: %s", target_str, exc)
        return []

    if gh.empty or ls.empty:
        return []

    gh = gh.drop_duplicates(subset=["GAME_ID"])
    scheduled_games: List[Dict[str, Any]] = []

    for _, row in gh.iterrows():
        gid = str(row["GAME_ID"])
        h_id = int(row["HOME_TEAM_ID"])
        a_id = int(row["VISITOR_TEAM_ID"])
        status = str(row.get("GAME_STATUS_TEXT", "Scheduled")).strip()

        teams_in_game = ls[ls["GAME_ID"] == gid]
        h_row = teams_in_game[teams_in_game["TEAM_ID"] == h_id]
        a_row = teams_in_game[teams_in_game["TEAM_ID"] == a_id]

        if not h_row.empty and not a_row.empty:
            h_abbrev = str(h_row.iloc[0]["TEAM_ABBREVIATION"])
            h_name = str(h_row.iloc[0].get("TEAM_NAME", h_abbrev))
            a_abbrev = str(a_row.iloc[0]["TEAM_ABBREVIATION"])
            a_name = str(a_row.iloc[0].get("TEAM_NAME", a_abbrev))

            label = f"{a_name} ({a_abbrev}) @ {h_name} ({h_abbrev}) — {status}"

            scheduled_games.append({
                "game_id": gid,
                "game_date": target_str,
                "home_team_id": h_id,
                "home_team_abbrev": h_abbrev,
                "home_abbrev": h_abbrev,
                "home_team_name": h_name,
                "home_name": h_name,
                "away_team_id": a_id,
                "away_team_abbrev": a_abbrev,
                "away_abbrev": a_abbrev,
                "away_team_name": a_name,
                "away_name": a_name,
                "game_status": status,
                "label": label,
            })

    _SCHEDULE_CACHE[target_str] = (now_ts, scheduled_games)
    return scheduled_games


def get_team_recent_rotation(
    team_abbrev: str,
    max_days_lookback: int = 45,
    min_games: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get active rotation players for a team based on their most recent appearances.
    Returns players sorted descending by average minutes played (starters first).
    """
    clean_p = get_player_logs()
    team_logs = clean_p[clean_p["TEAM_ABBREVIATION"] == team_abbrev].copy()
    if team_logs.empty:
        return []

    latest_date = team_logs["GAME_DATE"].max()
    cutoff_date = latest_date - pd.Timedelta(days=max_days_lookback)
    recent = team_logs[team_logs["GAME_DATE"] >= cutoff_date]

    if recent.empty:
        recent = team_logs.tail(150)

    stats = (
        recent.groupby(["PLAYER_ID", "PLAYER_NAME"])
        .agg(
            games_played=("GAME_ID", "count"),
            avg_minutes=("MIN", "mean"),
            avg_pts=("PTS", "mean"),
        )
        .reset_index()
    )

    stats = stats[stats["games_played"] >= min_games].sort_values("avg_minutes", ascending=False)

    rotation = []
    for _, r in stats.iterrows():
        rotation.append({
            "player_id": int(r["PLAYER_ID"]),
            "player_name": str(r["PLAYER_NAME"]),
            "name": str(r["PLAYER_NAME"]),
            "avg_minutes": round(float(r["avg_minutes"]), 1),
            "avg_min": round(float(r["avg_minutes"]), 1),
            "avg_pts": round(float(r["avg_pts"]), 1),
            "games_played": int(r["games_played"]),
            "display": f"{r['PLAYER_NAME']} ({r['avg_minutes']:.1f} mpg, {r['avg_pts']:.1f} ppg)",
        })

    return rotation
