"""Runner script for Gate 2: Deep Data Validation & Transactional SQL Ingestion."""

import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nba_assistant.stager import (
    stage_and_validate_team_logs,
    stage_and_validate_player_logs,
    run_full_sql_verification,
)


def main():
    start_time = time.time()
    print("=" * 70)
    print("GATE 2: DEEP DATA VALIDATION & TRANSACTIONAL SQL INGESTION")
    print("Database: NBA_Prediction (SQL Server LocalDB)")
    print("Coverage: 2014-15 to 2026-27 (13 Seasons)")
    print("Checks: Schema Contract, Null IDs, Statistical Sanity, Primary Key Duplicates")
    print("=" * 70)

    # 1. Team logs validation & staging
    team_results = stage_and_validate_team_logs()

    # 2. Player logs validation & staging
    player_results = stage_and_validate_player_logs()

    # 3. SQL verification
    sql_report = run_full_sql_verification()

    elapsed = round(time.time() - start_time, 2)
    summary = {
        "execution_time_seconds": elapsed,
        "team_logs": team_results,
        "player_logs": player_results,
        "sql_verification": sql_report,
    }

    report_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gate2_staging_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"GATE 2 EXECUTION COMPLETE IN {elapsed}s")
    print(f"Summary report written to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

