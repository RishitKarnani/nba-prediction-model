"""NBA API endpoint testing and schema investigation script.

Tests the 8 core endpoints requested in Task 2.1:
1. CommonAllPlayers (Active players identification)
2. PlayerGameLogs / PlayerGameLog
3. BoxScoreTraditionalV3
4. BoxScoreAdvancedV3
5. LeagueDashPlayerStats
6. LeagueDashTeamStats
7. TeamGameLog
8. CommonTeamRoster

Saves sample Parquet data to data/sample/ and produces schema/coverage metadata.
"""

import os
import json
import time
from typing import Dict, Any
import pandas as pd

from nba_api.stats.endpoints import (
    commonallplayers,
    playergamelog,
    teamgamelog,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
    leaguedashplayerstats,
    leaguedashteamstats,
    commonteamroster,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SLEEP_INTERVAL = 0.6  # Safe rate-limiting delay between requests


def test_endpoint(name: str, fetch_fn) -> Dict[str, Any]:
    """Execute endpoint call with error handling, timing, and schema inspection."""
    print(f"\n--- Testing Endpoint: {name} ---", flush=True)
    start = time.time()
    try:
        df = fetch_fn()
        elapsed = round(time.time() - start, 2)
        cols = list(df.columns)
        null_counts = df.isnull().sum().to_dict()
        sample_file = os.path.join(SAMPLE_DIR, f"{name.lower()}.parquet")
        df.to_parquet(sample_file, index=False)

        print(f"  [SUCCESS] Elapsed: {elapsed}s | Rows: {len(df)} | Columns: {len(cols)}", flush=True)
        print(f"  Saved sample to {sample_file}", flush=True)

        return {
            "endpoint": name,
            "status": "SUCCESS",
            "elapsed_seconds": elapsed,
            "row_count": len(df),
            "column_count": len(cols),
            "columns": cols,
            "null_summary": {k: int(v) for k, v in null_counts.items() if v > 0},
            "sample_record": df.head(1).to_dict(orient="records")[0] if not df.empty else {},
        }
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        print(f"  [FAILED] Elapsed: {elapsed}s | Error: {exc}", flush=True)
        return {
            "endpoint": name,
            "status": "FAILED",
            "elapsed_seconds": elapsed,
            "error": str(exc),
            "columns": [],
        }


def run_investigation():
    results = []

    # 1. CommonAllPlayers (Active players)
    def fetch_active_players():
        ep = commonallplayers.CommonAllPlayers(is_only_current_season=1, timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("CommonAllPlayers", fetch_active_players))

    sample_player_id = "203999"  # Nikola Jokic
    sample_team_id = "1610612743"  # Denver Nuggets

    # 2. PlayerGameLog
    def fetch_player_logs():
        ep = playergamelog.PlayerGameLog(player_id=sample_player_id, season="2023-24", timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    res_logs = test_endpoint("PlayerGameLog", fetch_player_logs)
    results.append(res_logs)

    # Pick Game ID from retrieved logs
    sample_game_id = "0022301180"

    # 3. BoxScoreTraditionalV3
    def fetch_box_trad():
        ep = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=sample_game_id, timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("BoxScoreTraditionalV3", fetch_box_trad))

    # 4. BoxScoreAdvancedV3
    def fetch_box_adv():
        ep = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=sample_game_id, timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("BoxScoreAdvancedV3", fetch_box_adv))

    # 5. LeagueDashPlayerStats
    def fetch_player_dash():
        ep = leaguedashplayerstats.LeagueDashPlayerStats(season="2023-24", per_mode_detailed="PerGame", timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("LeagueDashPlayerStats", fetch_player_dash))

    # 6. LeagueDashTeamStats
    def fetch_team_dash():
        ep = leaguedashteamstats.LeagueDashTeamStats(season="2023-24", per_mode_detailed="PerGame", timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("LeagueDashTeamStats", fetch_team_dash))

    # 7. TeamGameLog
    def fetch_team_logs():
        ep = teamgamelog.TeamGameLog(team_id=sample_team_id, season="2023-24", timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("TeamGameLog", fetch_team_logs))

    # 8. CommonTeamRoster
    def fetch_roster():
        ep = commonteamroster.CommonTeamRoster(team_id=sample_team_id, season="2023-24", timeout=20)
        time.sleep(SLEEP_INTERVAL)
        return ep.get_data_frames()[0]

    results.append(test_endpoint("CommonTeamRoster", fetch_roster))

    # Write JSON summary report
    report_file = os.path.join(SAMPLE_DIR, "endpoint_investigation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll endpoint investigations finished! Report: {report_file}", flush=True)


if __name__ == "__main__":
    run_investigation()
