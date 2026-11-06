"""Orchestration script for historical NBA data ingestion (2014-2024).

Implements the Hybrid Checkpoint Pipeline:
1. Download 10 seasons of team game logs to local Parquet cache with checkpointing.
2. Download 10 seasons of player game logs to local Parquet cache with checkpointing.
3. Ingest and stage all Parquet files into SQL Server (NBA_Prediction).
4. Run SQL verification queries and generate comprehensive summary report.
"""

import sys
import os
import json
import time

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nba_assistant.checkpoint import CheckpointManager
from nba_assistant.collector import download_historical_team_logs, download_historical_player_logs
from nba_assistant.stager import stage_team_logs_to_sql, stage_player_logs_to_sql, run_sql_verification

HISTORICAL_SEASONS = [
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]


def main():
    start_time = time.time()
    print("=" * 70)
    print("NBA HISTORICAL INGESTION PIPELINE: 2014-15 to 2023-24 (10 Seasons)")
    print("Architecture: Resilient Bulk Fetch -> Parquet Cache -> SQL Server LocalDB")
    print("=" * 70)

    checkpoint_mgr = CheckpointManager()

    # Step 1: Download Team Game Logs
    print("\n[PHASE 1/4] Collecting Team Game Logs...")
    team_results = download_historical_team_logs(HISTORICAL_SEASONS, checkpoint_mgr, delay_seconds=1.2)
    print(f"Team logs phase finished: {team_results['downloaded']} downloaded, {team_results['skipped']} skipped, {team_results['failed']} failed.")

    # Step 2: Download Player Game Logs
    print("\n[PHASE 2/4] Collecting Player Game Logs...")
    player_results = download_historical_player_logs(HISTORICAL_SEASONS, checkpoint_mgr, delay_seconds=1.2)
    print(f"Player logs phase finished: {player_results['downloaded']} downloaded, {player_results['skipped']} skipped, {player_results['failed']} failed.")

    # Step 3: Ingest Team & Player Logs into SQL Server
    print("\n[PHASE 3/4] Staging Data into SQL Server (NBA_Prediction)...")
    t_stage = stage_team_logs_to_sql()
    p_stage = stage_player_logs_to_sql()

    # Step 4: Verification Queries
    print("\n[PHASE 4/4] Verifying SQL Server Staging...")
    verification_report = run_sql_verification()

    elapsed = round(time.time() - start_time, 2)
    summary = {
        "execution_time_seconds": elapsed,
        "team_logs_staged": t_stage.get("loaded_rows", 0),
        "player_logs_staged": p_stage.get("loaded_rows", 0),
        "checkpoint_manifest": checkpoint_mgr.get_summary(),
        "sql_verification": verification_report,
    }

    # Save summary report
    summary_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ingestion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"HISTORICAL INGESTION COMPLETED IN {elapsed}s")
    print(f"Summary report written to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
