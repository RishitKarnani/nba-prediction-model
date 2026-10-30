"""Fetch multi-player, multi-season sample NBA data and stage it in SQL Server.

Scope:
- 3 Active Players: Nikola Jokić, LeBron James, Luka Dončić
- 2 Seasons: 2022-23, 2023-24
- Team logs & rosters: Denver Nuggets, Los Angeles Lakers
- Traditional & Advanced Box Scores for sample head-to-head games
- Ingest into staging tables in SQL Server database 'NBA_Prediction'
"""

import sys
import os
import time
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.connection import get_engine

from nba_api.stats.endpoints import (
    commonallplayers,
    playergamelog,
    teamgamelog,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
    commonteamroster,
)

SAMPLE_PLAYERS = [
    {"id": 203999, "name": "Nikola Jokic"},
    {"id": 2544, "name": "LeBron James"},
    {"id": 1629029, "name": "Luka Doncic"},
]

SAMPLE_TEAMS = [
    {"id": 1610612743, "name": "Denver Nuggets", "abbr": "DEN"},
    {"id": 1610612747, "name": "Los Angeles Lakers", "abbr": "LAL"},
]

SEASONS = ["2022-23", "2023-24"]
SLEEP_DELAY = 0.6


def fetch_and_stage():
    engine = get_engine()
    print("=" * 60)
    print("Starting NBA Sample Data Ingestion & SQL Server Staging")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Active Players Reference Data
    # ---------------------------------------------------------
    print("\n[1/5] Fetching Active Players Reference...")
    all_players_ep = commonallplayers.CommonAllPlayers(is_only_current_season=1, timeout=20)
    time.sleep(SLEEP_DELAY)
    df_players = all_players_ep.get_data_frames()[0]
    
    # Filter to selected players or keep all active players for reference
    df_players["ingested_at"] = pd.Timestamp.now()
    df_players.to_sql("stg_players", engine, if_exists="replace", index=False)
    print(f"  -> Staged {len(df_players)} players into [stg_players]")

    # ---------------------------------------------------------
    # 2. Player Game Logs (3 players x 2 seasons)
    # ---------------------------------------------------------
    print("\n[2/5] Fetching Player Game Logs for 3 players across 2 seasons...")
    player_logs_list = []
    for p in SAMPLE_PLAYERS:
        for s in SEASONS:
            print(f"  Fetching logs for {p['name']} ({s})...")
            ep = playergamelog.PlayerGameLog(player_id=p["id"], season=s, timeout=20)
            time.sleep(SLEEP_DELAY)
            df = ep.get_data_frames()[0]
            if not df.empty:
                df["SEASON"] = s
                df["PLAYER_NAME"] = p["name"]
                player_logs_list.append(df)

    if player_logs_list:
        df_player_logs = pd.concat(player_logs_list, ignore_index=True)
        # Drop duplicates on Player_ID and Game_ID
        df_player_logs = df_player_logs.drop_duplicates(subset=["Player_ID", "Game_ID"])
        df_player_logs["ingested_at"] = pd.Timestamp.now()
        df_player_logs.to_sql("stg_player_game_logs", engine, if_exists="replace", index=False)
        print(f"  -> Staged {len(df_player_logs)} player game log rows into [stg_player_game_logs]")
    else:
        df_player_logs = pd.DataFrame()

    # ---------------------------------------------------------
    # 3. Team Game Logs (2 teams x 2 seasons)
    # ---------------------------------------------------------
    print("\n[3/5] Fetching Team Game Logs for DEN and LAL across 2 seasons...")
    team_logs_list = []
    for t in SAMPLE_TEAMS:
        for s in SEASONS:
            print(f"  Fetching logs for {t['name']} ({s})...")
            ep = teamgamelog.TeamGameLog(team_id=t["id"], season=s, timeout=20)
            time.sleep(SLEEP_DELAY)
            df = ep.get_data_frames()[0]
            if not df.empty:
                df["SEASON"] = s
                df["TEAM_NAME"] = t["name"]
                team_logs_list.append(df)

    if team_logs_list:
        df_team_logs = pd.concat(team_logs_list, ignore_index=True)
        df_team_logs = df_team_logs.drop_duplicates(subset=["Team_ID", "Game_ID"])
        df_team_logs["ingested_at"] = pd.Timestamp.now()
        df_team_logs.to_sql("stg_team_game_logs", engine, if_exists="replace", index=False)
        print(f"  -> Staged {len(df_team_logs)} team game log rows into [stg_team_game_logs]")

    # ---------------------------------------------------------
    # 4. Box Scores Traditional & Advanced (Sample Games)
    # ---------------------------------------------------------
    print("\n[4/5] Fetching Traditional & Advanced Box Scores for sample games...")
    # Select 2 distinct Game IDs from the retrieved player logs
    sample_game_ids = df_player_logs["Game_ID"].drop_duplicates().head(2).tolist()
    
    trad_list = []
    adv_list = []
    for gid in sample_game_ids:
        print(f"  Fetching box scores for Game ID: {gid}...")
        ep_t = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=20)
        time.sleep(SLEEP_DELAY)
        df_t = ep_t.get_data_frames()[0]
        if not df_t.empty:
            trad_list.append(df_t)

        ep_a = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid, timeout=20)
        time.sleep(SLEEP_DELAY)
        df_a = ep_a.get_data_frames()[0]
        if not df_a.empty:
            adv_list.append(df_a)

    if trad_list:
        df_trad = pd.concat(trad_list, ignore_index=True)
        df_trad["ingested_at"] = pd.Timestamp.now()
        df_trad.to_sql("stg_boxscore_traditional", engine, if_exists="replace", index=False)
        print(f"  -> Staged {len(df_trad)} traditional box score rows into [stg_boxscore_traditional]")

    if adv_list:
        df_adv = pd.concat(adv_list, ignore_index=True)
        df_adv["ingested_at"] = pd.Timestamp.now()
        df_adv.to_sql("stg_boxscore_advanced", engine, if_exists="replace", index=False)
        print(f"  -> Staged {len(df_adv)} advanced box score rows into [stg_boxscore_advanced]")

    # ---------------------------------------------------------
    # 5. Team Rosters History (2 teams x 2 seasons)
    # ---------------------------------------------------------
    print("\n[5/5] Fetching Team Rosters for DEN and LAL across 2 seasons...")
    roster_list = []
    for t in SAMPLE_TEAMS:
        for s in SEASONS:
            print(f"  Fetching roster for {t['name']} ({s})...")
            ep = commonteamroster.CommonTeamRoster(team_id=t["id"], season=s, timeout=20)
            time.sleep(SLEEP_DELAY)
            df = ep.get_data_frames()[0]
            if not df.empty:
                df["SEASON"] = s
                df["TEAM_NAME"] = t["name"]
                roster_list.append(df)

    if roster_list:
        df_roster = pd.concat(roster_list, ignore_index=True)
        df_roster = df_roster.drop_duplicates(subset=["TeamID", "PLAYER_ID", "SEASON"])
        df_roster["ingested_at"] = pd.Timestamp.now()
        df_roster.to_sql("stg_team_rosters", engine, if_exists="replace", index=False)
        print(f"  -> Staged {len(df_roster)} roster association rows into [stg_team_rosters]")

    # ---------------------------------------------------------
    # Verification Queries
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running SQL Server Verification Queries in NBA_Prediction")
    print("=" * 60)
    with engine.connect() as conn:
        tables = [
            "stg_players",
            "stg_player_game_logs",
            "stg_team_game_logs",
            "stg_boxscore_traditional",
            "stg_boxscore_advanced",
            "stg_team_rosters",
        ]
        for tbl in tables:
            cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            print(f"  Table [{tbl}]: {cnt} records")

        # Sample query on player game logs
        print("\nTop 3 Jokic Games by Points (from SQL Server):")
        query = text("""
            SELECT TOP 3 SEASON, GAME_DATE, MATCHUP, MIN, PTS, REB, AST, STL, BLK, TOV, FG_PCT
            FROM stg_player_game_logs
            WHERE PLAYER_NAME = 'Nikola Jokic'
            ORDER BY PTS DESC
        """)
        for row in conn.execute(query):
            print(f"  {row.SEASON} | {row.GAME_DATE} | {row.MATCHUP} | PTS: {row.PTS} | REB: {row.REB} | AST: {row.AST} | STL: {row.STL} | BLK: {row.BLK}")

    print("\nIngestion and SQL staging successfully completed!")


if __name__ == "__main__":
    fetch_and_stage()
