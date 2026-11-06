"""Script to download 10 seasons (2014-15 to 2023-24) of raw NBA data to Parquet.

Gate 1 Lightweight Validation:
- Non-empty response (>0 rows)
- Presence of essential IDs (GAME_ID, TEAM_ID, PLAYER_ID)
- Snappy-compressed Parquet storage in data/raw/
- Checkpoint registry logging in data/raw/ingestion_manifest.json
"""

import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nba_assistant.checkpoint import CheckpointManager
from nba_assistant.collector import download_historical_team_logs, download_historical_player_logs

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
    "2024-25",
    "2025-26",
    "2026-27",
]


def main():
    start_time = time.time()
    print("=" * 70)
    print("NBA HISTORICAL RAW DATA DOWNLOAD: 2014-15 to 2023-24 (10 Seasons)")
    print("Target: data/raw/team_logs & data/raw/player_logs (Parquet Snappy)")
    print("Validation: Gate 1 (Non-empty + Core Identifier Contract)")
    print("=" * 70)

    checkpoint_mgr = CheckpointManager()

    # Step 1: Download Team Game Logs
    team_results = download_historical_team_logs(HISTORICAL_SEASONS, checkpoint_mgr, delay_seconds=1.5)

    # Step 2: Download Player Game Logs
    player_results = download_historical_player_logs(HISTORICAL_SEASONS, checkpoint_mgr, delay_seconds=1.5)

    elapsed = round(time.time() - start_time, 2)
    summary = {
        "execution_time_seconds": elapsed,
        "team_logs": {
            "downloaded": team_results["downloaded"],
            "skipped": team_results["skipped"],
            "failed": team_results["failed"],
            "total_rows": team_results["total_rows"],
        },
        "player_logs": {
            "downloaded": player_results["downloaded"],
            "skipped": player_results["skipped"],
            "failed": player_results["failed"],
            "total_rows": player_results["total_rows"],
        },
        "checkpoint_manifest": checkpoint_mgr.get_summary(),
    }

    report_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gate1_download_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"STEP 1 COMPLETE: Raw Data Downloaded and Cached in {elapsed}s")
    print(f"Team Logs: {team_results['total_rows']:,} rows across {len(team_results['files'])} files")
    print(f"Player Logs: {player_results['total_rows']:,} rows across {len(player_results['files'])} files")
    print(f"Summary written to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

