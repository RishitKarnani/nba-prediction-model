"""
src/roster_minutes_historical.py
────────────────────────────────
Extracts active 15-man rosters for all 30 NBA teams for a specified historical season
(e.g., 2024-25) and predicts their playing time exclusively using the preserved
LightGBM Minutes Prediction Model.

Ensures proper handling of player movement, trades, and roster turnover by strictly
slicing rosters as they existed in that historical season.
Normalizes individual predicted minutes so that sum(MIN) == 240.0.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.roster_minutes import extract_player_minute_features, load_minutes_model
from src.tactical_registry import TEAM_TACTICAL_ACTIONS

logger = logging.getLogger(__name__)


def get_historical_player_logs(season: str = "2024-25") -> pd.DataFrame:
    """Load player game logs for a specific historical season."""
    path = ROOT_DIR / "data" / "raw" / "player_logs" / f"player_logs_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Player logs for season {season} not found at {path}")

    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0.0)
    df["PLUS_MINUS"] = pd.to_numeric(df.get("PLUS_MINUS", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    return df


def generate_historical_team_predicted_minutes(
    season: str = "2024-25",
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate 15-man active rosters and model-predicted minutes for a historical season.
    """
    cache_path = ROOT_DIR / "artefacts" / f"roster_predicted_minutes_{season.replace('-', '_')}.json"
    if not force_refresh and cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info(f"Generating historical predicted minutes for {season} season across all 30 teams...")
    model = load_minutes_model()
    feature_names = list(model.feature_name_)
    df = get_historical_player_logs(season)

    team_rotations: Dict[str, Dict[str, Any]] = {}

    for abbrev in TEAM_TACTICAL_ACTIONS.keys():
        team_df = df[df["TEAM_ABBREVIATION"] == abbrev]
        if team_df.empty:
            continue

        player_counts = (
            team_df.groupby(["PLAYER_ID", "PLAYER_NAME"])
            .agg(games=("GAME_ID", "count"), total_min=("MIN", "sum"), avg_min=("MIN", "mean"))
            .reset_index()
            .sort_values(["games", "total_min"], ascending=[False, False])
        )

        top_15_records = player_counts.head(15).to_dict(orient="records")
        predicted_players = []
        raw_pred_mins = []

        for p in top_15_records:
            p_id = p["PLAYER_ID"]
            p_name = p["PLAYER_NAME"]
            p_df = team_df[team_df["PLAYER_ID"] == p_id]

            feat_dict = extract_player_minute_features(p_id, p_df, feature_names)
            feat_df = pd.DataFrame([feat_dict])
            raw_min = float(model.predict(feat_df)[0])
            raw_min = float(np.clip(raw_min, 1.0, 42.0))

            raw_pred_mins.append(raw_min)
            predicted_players.append({
                "player_id": int(p_id),
                "player_name": str(p_name),
                "raw_predicted_minutes": round(raw_min, 2),
                "historical_games": int(p["games"]),
                "historical_pts_per_game": round(float(p_df["PTS"].mean()), 1) if not p_df.empty else 0.0,
                "historical_plus_minus": round(float(p_df["PLUS_MINUS"].mean()), 2) if not p_df.empty else 0.0,
            })

        # 240-Minute Normalization Invariant
        sum_pred = sum(raw_pred_mins)
        if sum_pred <= 0:
            sum_pred = 240.0

        for p_dict, raw_min in zip(predicted_players, raw_pred_mins):
            ratio = raw_min / sum_pred
            allocated_min = round(240.0 * ratio, 2)
            p_dict["minute_ratio"] = round(ratio, 4)
            p_dict["allocated_minutes"] = allocated_min

        predicted_players.sort(key=lambda x: x["allocated_minutes"], reverse=True)

        for idx, p in enumerate(predicted_players):
            if idx < 5:
                p["role"] = "Starter"
            elif idx < 9:
                p["role"] = "Key Bench"
            elif idx < 12:
                p["role"] = "Situational"
            else:
                p["role"] = "Deep Bench"

        team_rotations[abbrev] = {
            "season": season,
            "team_abbreviation": abbrev,
            "team_name": TEAM_TACTICAL_ACTIONS[abbrev]["team_name"],
            "total_allocated_minutes": round(sum(p["allocated_minutes"] for p in predicted_players), 2),
            "lineup": predicted_players,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(team_rotations, f, indent=2)

    logger.info(f"Successfully generated and cached {season} rosters and predicted minutes.")
    return team_rotations


if __name__ == "__main__":
    rot = generate_historical_team_predicted_minutes("2024-25", force_refresh=True)
    print(f"Generated 2024-25 rosters for {len(rot)} teams.")
    for tm in ["BOS", "DAL", "MIN"]:
        print(f"\n{rot[tm]['team_name']} (2024-25 Roster - Top 5 Minutes):")
        for p in rot[tm]["lineup"][:5]:
            print(f"  * {p['player_name']:<24} [{p['role']:<9}] Pred Min: {p['allocated_minutes']:>4.1f}m ({p['minute_ratio']*100:.1f}%)")
