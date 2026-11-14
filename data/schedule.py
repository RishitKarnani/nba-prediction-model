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


def get_daily_schedule(game_date: Optional[str | datetime.date] = None) -> List[Dict[str, Any]]:
    """
    Fetch the official NBA schedule for a specific date using nba_api.

    Parameters
    ----------
    game_date : 'YYYY-MM-DD' or datetime.date. Defaults to today's date.

    Returns
    -------
    List of game dicts with keys:
        game_id, game_date, home_team_id, home_team_abbrev, home_team_name,
        away_team_id, away_team_abbrev, away_team_name, game_status, label
    """
    if game_date is None:
        target_str = datetime.date.today().strftime("%Y-%m-%d")
    elif isinstance(game_date, datetime.date):
        target_str = game_date.strftime("%Y-%m-%d")
    else:
        target_str = str(game_date).strip()

    try:
        sb = scoreboardv2.ScoreboardV2(game_date=target_str)
        gh = sb.game_header.get_data_frame()
        ls = sb.line_score.get_data_frame()
    except Exception as exc:
        logger.warning("Failed to fetch live Scoreboard for %s: %s", target_str, exc)
        return []

    if gh.empty or ls.empty:
        return []

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

            label = f"{a_abbrev} @ {h_abbrev} ({status})"

            scheduled_games.append({
                "game_id": gid,
                "game_date": target_str,
                "home_team_id": h_id,
                "home_team_abbrev": h_abbrev,
                "home_team_name": h_name,
                "away_team_id": a_id,
                "away_team_abbrev": a_abbrev,
                "away_team_name": a_name,
                "game_status": status,
                "label": label,
            })

    return scheduled_games


def get_team_recent_rotation(
    team_abbrev: str,
    max_days_lookback: int = 30,
    min_games: int = 2,
) -> List[Dict[str, Any]]:
    """
    Get active rotation players for a team based on their most recent historical appearances.
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
        recent = team_logs.tail(100)

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
            "avg_minutes": round(float(r["avg_minutes"]), 1),
            "avg_pts": round(float(r["avg_pts"]), 1),
            "games_played": int(r["games_played"]),
            "display": f"{r['PLAYER_NAME']} ({r['avg_minutes']:.1f} mpg, {r['avg_pts']:.1f} ppg)",
        })

    return rotation
