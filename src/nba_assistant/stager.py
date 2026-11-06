"""Transactional, idempotent SQL Server staging module with Gate 2 validation."""

import os
from typing import Dict, Any, List
import pandas as pd
from sqlalchemy import text

from database.connection import get_engine
from nba_assistant.schema_registry import validate_schema, TEAM_LOG_SCHEMA, PLAYER_LOG_SCHEMA
from nba_assistant.validator import validate_team_logs_data, validate_player_logs_data

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def stage_and_validate_team_logs() -> Dict[str, Any]:
    """Validate and transactionally stage all team game logs Parquet files into SQL Server."""
    engine = get_engine()
    team_dir = os.path.join(RAW_DIR, "team_logs")
    parquet_files = sorted([os.path.join(team_dir, f) for f in os.listdir(team_dir) if f.endswith(".parquet")])

    print(f"\n=== Gate 2: Team Game Logs Validation & Transactional Staging ({len(parquet_files)} seasons) ===")
    results = {"total_staged": 0, "seasons_processed": [], "validation_passed": 0}

    for idx, pf in enumerate(parquet_files, 1):
        season = os.path.basename(pf).replace("team_logs_", "").replace(".parquet", "")
        df = pd.read_parquet(pf)

        # 1. Schema Registry Contract Validation
        validate_schema(df, TEAM_LOG_SCHEMA)

        # 2. Deep Data Validation (Null IDs, Impossible Stats, Row Bounds)
        val_report = validate_team_logs_data(df, season)
        results["validation_passed"] += 1

        # 3. Transactional, Idempotent Loading
        # Delete existing season partition and insert inside an atomic database transaction
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM dbo.stg_team_game_logs WHERE SEASON = :season"),
                {"season": season},
            )
            df["ingested_at"] = pd.Timestamp.now()
            df.to_sql(
                "stg_team_game_logs",
                conn,
                if_exists="append",
                index=False,
                chunksize=5000,
            )

        results["total_staged"] += len(df)
        results["seasons_processed"].append({"season": season, "rows": len(df)})
        print(f"  [{idx}/{len(parquet_files)}] Season {season}: {len(df):,} rows validated & staged into SQL Server.")

    print(f"  [SUCCESS] All {len(parquet_files)} team seasons validated and staged ({results['total_staged']:,} rows).")
    return results


def stage_and_validate_player_logs() -> Dict[str, Any]:
    """Validate and transactionally stage all player game logs Parquet files into SQL Server."""
    engine = get_engine()
    player_dir = os.path.join(RAW_DIR, "player_logs")
    parquet_files = sorted([os.path.join(player_dir, f) for f in os.listdir(player_dir) if f.endswith(".parquet")])

    print(f"\n=== Gate 2: Player Game Logs Validation & Transactional Staging ({len(parquet_files)} seasons) ===")
    results = {"total_staged": 0, "seasons_processed": [], "validation_passed": 0}

    for idx, pf in enumerate(parquet_files, 1):
        season = os.path.basename(pf).replace("player_logs_", "").replace(".parquet", "")
        df = pd.read_parquet(pf)

        # 1. Schema Registry Contract Validation
        validate_schema(df, PLAYER_LOG_SCHEMA)

        # 2. Deep Data Validation (Null IDs, Impossible Stats, Row Bounds)
        val_report = validate_player_logs_data(df, season)
        results["validation_passed"] += 1

        # 3. Transactional, Idempotent Loading
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM dbo.stg_player_game_logs WHERE SEASON = :season"),
                {"season": season},
            )
            df["ingested_at"] = pd.Timestamp.now()
            df.to_sql(
                "stg_player_game_logs",
                conn,
                if_exists="append",
                index=False,
                chunksize=5000,
            )

        results["total_staged"] += len(df)
        results["seasons_processed"].append({"season": season, "rows": len(df)})
        print(f"  [{idx}/{len(parquet_files)}] Season {season}: {len(df):,} rows validated & staged into SQL Server.")

    print(f"  [SUCCESS] All {len(parquet_files)} player seasons validated and staged ({results['total_staged']:,} rows).")
    return results


def run_full_sql_verification() -> Dict[str, Any]:
    """Execute rigorous SQL Server verification queries to prove data integrity."""
    engine = get_engine()
    report = {}
    print("\n" + "=" * 70)
    print("RUNNING FINAL SQL SERVER VERIFICATION IN NBA_Prediction")
    print("=" * 70)

    with engine.connect() as conn:
        # 1. Team Game Logs Summary
        t_summary = conn.execute(
            text("""
            SELECT 
                COUNT(*) AS total_rows,
                COUNT(DISTINCT GAME_ID) AS unique_games,
                COUNT(DISTINCT TEAM_ID) AS unique_teams,
                MIN(GAME_DATE) AS min_date,
                MAX(GAME_DATE) AS max_date,
                COUNT(DISTINCT SEASON) AS season_count
            FROM dbo.stg_team_game_logs;
        """)
        ).fetchone()

        # Check for duplicates on primary key
        t_dups = conn.execute(
            text("""
            SELECT COUNT(*) FROM (
                SELECT TEAM_ID, GAME_ID, COUNT(*) AS cnt
                FROM dbo.stg_team_game_logs
                GROUP BY TEAM_ID, GAME_ID
                HAVING COUNT(*) > 1
            ) d;
        """)
        ).scalar()

        report["team_logs"] = {
            "total_rows": t_summary.total_rows,
            "unique_games": t_summary.unique_games,
            "unique_teams": t_summary.unique_teams,
            "earliest_game": str(t_summary.min_date),
            "latest_game": str(t_summary.max_date),
            "seasons": t_summary.season_count,
            "duplicate_primary_keys": t_dups,
        }

        print("  [TEAM GAME LOGS]:")
        print(f"    Total Rows: {t_summary.total_rows:,}")
        print(f"    Unique Games: {t_summary.unique_games:,}")
        print(f"    Unique Teams: {t_summary.unique_teams}")
        print(f"    Date Range: {t_summary.min_date} to {t_summary.max_date}")
        print(f"    Primary Key Duplicates: {t_dups} (Strictly 0 expected)")

        # 2. Player Game Logs Summary
        p_summary = conn.execute(
            text("""
            SELECT 
                COUNT(*) AS total_rows,
                COUNT(DISTINCT PLAYER_ID) AS unique_players,
                COUNT(DISTINCT GAME_ID) AS unique_games,
                MIN(GAME_DATE) AS min_date,
                MAX(GAME_DATE) AS max_date,
                COUNT(DISTINCT SEASON) AS season_count
            FROM dbo.stg_player_game_logs;
        """)
        ).fetchone()

        p_dups = conn.execute(
            text("""
            SELECT COUNT(*) FROM (
                SELECT PLAYER_ID, GAME_ID, COUNT(*) AS cnt
                FROM dbo.stg_player_game_logs
                GROUP BY PLAYER_ID, GAME_ID
                HAVING COUNT(*) > 1
            ) d;
        """)
        ).scalar()

        report["player_logs"] = {
            "total_rows": p_summary.total_rows,
            "unique_players": p_summary.unique_players,
            "unique_games": p_summary.unique_games,
            "earliest_game": str(p_summary.min_date),
            "latest_game": str(p_summary.max_date),
            "seasons": p_summary.season_count,
            "duplicate_primary_keys": p_dups,
        }

        print("\n  [PLAYER GAME LOGS]:")
        print(f"    Total Rows: {p_summary.total_rows:,}")
        print(f"    Unique Players: {p_summary.unique_players:,}")
        print(f"    Unique Games: {p_summary.unique_games:,}")
        print(f"    Date Range: {p_summary.min_date} to {p_summary.max_date}")
        print(f"    Primary Key Duplicates: {p_dups} (Strictly 0 expected)")

        # 3. Sample verification query: Top 5 team scoring games across modern era
        print("\n  [SAMPLE SQL QUERY - Top 5 Team Scoring Games (2014-2026)]:")
        top_games = conn.execute(
            text("""
            SELECT TOP 5 SEASON, GAME_DATE, TEAM_NAME, MATCHUP, WL, PTS, FGM, FGA, FG_PCT, FG3M
            FROM dbo.stg_team_game_logs
            ORDER BY PTS DESC;
        """)
        ).fetchall()
        for g in top_games:
            print(f"    {g.SEASON} | {g.GAME_DATE} | {g.TEAM_NAME} ({g.MATCHUP}) | PTS: {g.PTS} | FG%: {g.FG_PCT:.3f} | 3PM: {g.FG3M}")

    return report
