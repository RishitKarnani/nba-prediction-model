"""Resilient NBA historical data collector with Gate 1 lightweight checks and checkpointing."""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Callable
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from nba_assistant.checkpoint import CheckpointManager

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Configuration parameters
DEFAULT_RATE_LIMIT_DELAY = 1.5
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
INITIAL_BACKOFF = 3.0

# Gate 1 Minimal Identifier Contracts
TEAM_CORE_COLUMNS = {"GAME_ID", "TEAM_ID", "GAME_DATE", "WL", "PTS"}
PLAYER_CORE_COLUMNS = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE", "PTS"}


def _fetch_with_retry(fetch_fn: Callable[[], pd.DataFrame], task_name: str) -> pd.DataFrame:
    """Execute fetch function with exponential backoff on network/rate-limit errors."""
    delay = INITIAL_BACKOFF
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fetch_fn()
            return df
        except Exception as exc:
            last_err = exc
            print(f"    [Attempt {attempt}/{MAX_RETRIES} failed for {task_name}]: {exc}")
            if attempt == MAX_RETRIES:
                break
            print(f"    Backoff: waiting {delay:.1f}s before retry...")
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {task_name}: {last_err}")


def _validate_gate_1(df: pd.DataFrame, expected_core_cols: set, task_name: str) -> None:
    """Gate 1 lightweight validation: ensures non-empty data and presence of core identifiers."""
    if df is None or df.empty:
        raise ValueError(f"Gate 1 Failed for {task_name}: Received empty DataFrame from API.")
    missing_cols = expected_core_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Gate 1 Failed for {task_name}: Missing core columns {missing_cols}.")


def download_historical_team_logs(
    seasons: List[str],
    checkpoint_mgr: CheckpointManager,
    delay_seconds: float = DEFAULT_RATE_LIMIT_DELAY,
) -> Dict[str, Any]:
    """Download team game logs for all specified seasons with Gate 1 checks."""
    out_dir = os.path.join(RAW_DIR, "team_logs")
    os.makedirs(out_dir, exist_ok=True)
    results = {"downloaded": 0, "skipped": 0, "failed": 0, "files": [], "total_rows": 0}

    print(f"\n=== Downloading Team Game Logs ({len(seasons)} seasons) ===")
    for season in seasons:
        task_key = f"team_logs_{season}"
        file_path = os.path.join(out_dir, f"team_logs_{season}.parquet")

        # Checkpoint check
        if checkpoint_mgr.is_completed(task_key) and os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            rows = len(existing_df)
            print(f"  [CHECKPOINT HIT] {season}: Already saved ({rows:,} rows). Skipping.")
            results["skipped"] += 1
            results["files"].append(file_path)
            results["total_rows"] += rows
            continue

        print(f"  Fetching team logs for season {season}...")
        start_t = time.time()
        try:
            def fetch():
                ep = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star="Regular Season",
                    player_or_team_abbreviation="T",
                    timeout=TIMEOUT_SECONDS,
                )
                return ep.get_data_frames()[0]

            df = _fetch_with_retry(fetch, task_key)

            # Gate 1 Check
            _validate_gate_1(df, TEAM_CORE_COLUMNS, task_key)

            # Metadata enrichment
            df["SEASON"] = season
            df["downloaded_at"] = datetime.now().isoformat()

            # Save to Parquet
            df.to_parquet(file_path, index=False, engine="pyarrow", compression="snappy")
            elapsed = round(time.time() - start_t, 2)

            checkpoint_mgr.record_success(
                task_key,
                file_path,
                len(df),
                {"season": season, "type": "team_logs", "columns": len(df.columns)},
            )
            print(f"  [GATE 1 PASSED] {season}: Saved {len(df):,} rows to {os.path.basename(file_path)} ({elapsed}s)")
            results["downloaded"] += 1
            results["files"].append(file_path)
            results["total_rows"] += len(df)
        except Exception as err:
            print(f"  [FAILED] {season}: Error: {err}")
            checkpoint_mgr.record_failure(task_key, str(err))
            results["failed"] += 1

        time.sleep(delay_seconds)

    return results


def download_historical_player_logs(
    seasons: List[str],
    checkpoint_mgr: CheckpointManager,
    delay_seconds: float = DEFAULT_RATE_LIMIT_DELAY,
) -> Dict[str, Any]:
    """Download player game logs for all specified seasons with Gate 1 checks."""
    out_dir = os.path.join(RAW_DIR, "player_logs")
    os.makedirs(out_dir, exist_ok=True)
    results = {"downloaded": 0, "skipped": 0, "failed": 0, "files": [], "total_rows": 0}

    print(f"\n=== Downloading Player Game Logs ({len(seasons)} seasons) ===")
    for season in seasons:
        task_key = f"player_logs_{season}"
        file_path = os.path.join(out_dir, f"player_logs_{season}.parquet")

        # Checkpoint check
        if checkpoint_mgr.is_completed(task_key) and os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            rows = len(existing_df)
            print(f"  [CHECKPOINT HIT] {season}: Already saved ({rows:,} rows). Skipping.")
            results["skipped"] += 1
            results["files"].append(file_path)
            results["total_rows"] += rows
            continue

        print(f"  Fetching player logs for season {season}...")
        start_t = time.time()
        try:
            def fetch():
                ep = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star="Regular Season",
                    player_or_team_abbreviation="P",
                    timeout=TIMEOUT_SECONDS,
                )
                return ep.get_data_frames()[0]

            df = _fetch_with_retry(fetch, task_key)

            # Gate 1 Check
            _validate_gate_1(df, PLAYER_CORE_COLUMNS, task_key)

            # Metadata enrichment
            df["SEASON"] = season
            df["downloaded_at"] = datetime.now().isoformat()

            # Save to Parquet
            df.to_parquet(file_path, index=False, engine="pyarrow", compression="snappy")
            elapsed = round(time.time() - start_t, 2)

            checkpoint_mgr.record_success(
                task_key,
                file_path,
                len(df),
                {"season": season, "type": "player_logs", "columns": len(df.columns)},
            )
            print(f"  [GATE 1 PASSED] {season}: Saved {len(df):,} rows to {os.path.basename(file_path)} ({elapsed}s)")
            results["downloaded"] += 1
            results["files"].append(file_path)
            results["total_rows"] += len(df)
        except Exception as err:
            print(f"  [FAILED] {season}: Error: {err}")
            checkpoint_mgr.record_failure(task_key, str(err))
            results["failed"] += 1

        time.sleep(delay_seconds)

    return results
