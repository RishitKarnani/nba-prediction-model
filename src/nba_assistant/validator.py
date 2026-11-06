"""Deep data validation module for Gate 2 verification before SQL loading."""

from typing import Dict, Any, List
import pandas as pd


class DataValidationError(Exception):
    """Raised when data values violate physical, logical, or integrity constraints."""
    pass


def validate_team_logs_data(df: pd.DataFrame, season: str) -> Dict[str, Any]:
    """Execute deep integrity and statistical sanity checks on team game logs."""
    report = {"season": season, "checks_passed": [], "failures": []}

    # 1. Null check on critical keys
    null_keys = {}
    cols_to_check = ["TEAM_ID", "GAME_ID", "GAME_DATE", "PTS"]
    # WL is mandatory for completed historical seasons; ongoing season may contain in-progress games
    if season != "2026-27":
        cols_to_check.append("WL")

    for col in cols_to_check:
        n = df[col].isnull().sum()
        if n > 0:
            null_keys[col] = int(n)

    if null_keys:
        msg = f"Null values found in critical keys: {null_keys}"
        report["failures"].append(msg)
        raise DataValidationError(f"[Team Logs {season}] {msg}")
    report["checks_passed"].append("Critical Keys Null Check (0 nulls)")

    # 2. Duplicate Natural Key Check (TEAM_ID, GAME_ID)
    dups = df.duplicated(subset=["TEAM_ID", "GAME_ID"]).sum()
    if dups > 0:
        msg = f"Found {dups} duplicate (TEAM_ID, GAME_ID) records"
        report["failures"].append(msg)
        raise DataValidationError(f"[Team Logs {season}] {msg}")
    report["checks_passed"].append(f"Uniqueness Check (0 duplicates across {len(df):,} rows)")

    # 3. Impossible Statistics Checks
    stat_violations = []
    if (df["PTS"] < 0).any():
        stat_violations.append("Negative PTS detected")
    if (df["MIN"] < 0).any():
        stat_violations.append("Negative MIN detected")

    if (df["FGM"] > df["FGA"]).any():
        stat_violations.append("FGM > FGA detected")
    if (df["FG3M"] > df["FG3A"]).any():
        stat_violations.append("FG3M > FG3A detected")
    if (df["FTM"] > df["FTA"]).any():
        stat_violations.append("FTM > FTA detected")

    for pct_col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
        valid_vals = df[pct_col].dropna()
        if ((valid_vals < 0.0) | (valid_vals > 1.0)).any():
            stat_violations.append(f"{pct_col} outside [0.0, 1.0]")

    for count_col in ["REB", "AST", "STL", "BLK", "TOV", "PF"]:
        if (df[count_col] < 0).any():
            stat_violations.append(f"Negative {count_col} detected")

    if stat_violations:
        msg = f"Impossible statistics detected: {stat_violations}"
        report["failures"].append(msg)
        raise DataValidationError(f"[Team Logs {season}] {msg}")
    report["checks_passed"].append("Impossible Statistics Check (All values within valid physical bounds)")

    # 4. Row Count Plausibility Bounds
    if season == "2026-27":
        min_bound, max_bound = 1, 2460
    elif season in ["2019-20", "2020-21"]:
        min_bound, max_bound = 2000, 2460
    else:
        min_bound, max_bound = 2400, 2460

    if not (min_bound <= len(df) <= max_bound):
        msg = f"Row count {len(df)} outside expected bounds [{min_bound}, {max_bound}]"
        report["failures"].append(msg)
        raise DataValidationError(f"[Team Logs {season}] {msg}")
    report["checks_passed"].append(f"Row Count Bounds Check ({len(df):,} within [{min_bound}, {max_bound}])")

    return report


def validate_player_logs_data(df: pd.DataFrame, season: str) -> Dict[str, Any]:
    """Execute deep integrity and statistical sanity checks on player game logs."""
    report = {"season": season, "checks_passed": [], "failures": []}

    # 1. Null check on critical keys
    null_keys = {}
    cols_to_check = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "PTS"]
    if season != "2026-27":
        cols_to_check.append("WL")

    for col in cols_to_check:
        n = df[col].isnull().sum()
        if n > 0:
            null_keys[col] = int(n)

    if null_keys:
        msg = f"Null values found in critical keys: {null_keys}"
        report["failures"].append(msg)
        raise DataValidationError(f"[Player Logs {season}] {msg}")
    report["checks_passed"].append("Critical Keys Null Check (0 nulls)")

    # 2. Duplicate Natural Key Check (PLAYER_ID, GAME_ID)
    dups = df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum()
    if dups > 0:
        msg = f"Found {dups} duplicate (PLAYER_ID, GAME_ID) records"
        report["failures"].append(msg)
        raise DataValidationError(f"[Player Logs {season}] {msg}")
    report["checks_passed"].append(f"Uniqueness Check (0 duplicates across {len(df):,} rows)")

    # 3. Impossible Statistics Checks
    stat_violations = []
    if (df["PTS"] < 0).any():
        stat_violations.append("Negative PTS detected")
    if (df["MIN"] < 0).any():
        stat_violations.append("Negative MIN detected")

    if (df["FGM"] > df["FGA"]).any():
        stat_violations.append("FGM > FGA detected")
    if (df["FG3M"] > df["FG3A"]).any():
        stat_violations.append("FG3M > FG3A detected")
    if (df["FTM"] > df["FTA"]).any():
        stat_violations.append("FTM > FTA detected")

    for pct_col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
        valid_vals = df[pct_col].dropna()
        if ((valid_vals < 0.0) | (valid_vals > 1.0)).any():
            stat_violations.append(f"{pct_col} outside [0.0, 1.0]")

    for count_col in ["REB", "AST", "STL", "BLK", "TOV", "PF"]:
        if (df[count_col] < 0).any():
            stat_violations.append(f"Negative {count_col} detected")

    if stat_violations:
        msg = f"Impossible statistics detected: {stat_violations}"
        report["failures"].append(msg)
        raise DataValidationError(f"[Player Logs {season}] {msg}")
    report["checks_passed"].append("Impossible Statistics Check (All values within valid physical bounds)")

    # 4. Row Count Plausibility Bounds
    if season == "2026-27":
        min_bound, max_bound = 500, 30000
    elif season in ["2019-20", "2020-21"]:
        min_bound, max_bound = 20000, 30000
    else:
        min_bound, max_bound = 24000, 30000

    if not (min_bound <= len(df) <= max_bound):
        msg = f"Row count {len(df)} outside expected bounds [{min_bound}, {max_bound}]"
        report["failures"].append(msg)
        raise DataValidationError(f"[Player Logs {season}] {msg}")
    report["checks_passed"].append(f"Row Count Bounds Check ({len(df):,} within [{min_bound}, {max_bound}])")

    return report
