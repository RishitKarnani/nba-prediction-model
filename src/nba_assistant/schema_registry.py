"""Schema registry defining expected schemas, columns, and data types for NBA datasets."""

from typing import Dict, Any, List, Set
import pandas as pd


class SchemaValidationError(Exception):
    """Raised when an incoming dataset violates the schema contract."""
    pass


TEAM_LOG_SCHEMA = {
    "name": "team_game_logs",
    "primary_keys": ["TEAM_ID", "GAME_ID"],
    "required_columns": [
        "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
        "SEASON"
    ],
    "numeric_columns": [
        "TEAM_ID", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS", "PLUS_MINUS"
    ],
}

PLAYER_LOG_SCHEMA = {
    "name": "player_game_logs",
    "primary_keys": ["PLAYER_ID", "GAME_ID"],
    "required_columns": [
        "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
        "SEASON"
    ],
    "numeric_columns": [
        "PLAYER_ID", "TEAM_ID", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS", "PLUS_MINUS"
    ],
}


def validate_schema(df: pd.DataFrame, schema_contract: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DataFrame against a defined schema contract. Detect unexpected drift."""
    contract_name = schema_contract["name"]
    missing_cols = [c for c in schema_contract["required_columns"] if c not in df.columns]
    if missing_cols:
        raise SchemaValidationError(f"[{contract_name}] Missing required columns: {missing_cols}")

    # Check numeric columns conversion validity
    non_numeric_errors = []
    for col in schema_contract["numeric_columns"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Attempt coercion check
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.isnull().sum() > df[col].isnull().sum():
                    non_numeric_errors.append(col)

    if non_numeric_errors:
        raise SchemaValidationError(f"[{contract_name}] Columns contain unparseable non-numeric values: {non_numeric_errors}")

    return {
        "status": "PASSED",
        "contract": contract_name,
        "columns_checked": len(schema_contract["required_columns"]),
        "observed_columns": len(df.columns),
    }

