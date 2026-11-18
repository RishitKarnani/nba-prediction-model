"""
src/roster_minutes.py
─────────────────────
Extracts active 15-man rosters for all 30 NBA teams and predicts their playing time
exclusively using the preserved LightGBM Minutes Prediction Model.

Normalizes individual player predicted minutes so that each team's rotation
strictly satisfies the regulation 240-minute invariant: sum(MIN) == 240.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import glob
import json
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.tactical_registry import TEAM_TACTICAL_ACTIONS

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MINUTES_MODEL_PATH = ROOT_DIR / "artefacts" / "models" / "minutes" / "minutes_model.joblib"
CACHE_PATH = ROOT_DIR / "artefacts" / "roster_predicted_minutes.json"


def load_minutes_model():
    """Load the trained LightGBM minutes prediction model."""
    if not MINUTES_MODEL_PATH.exists():
        raise FileNotFoundError(f"Minutes model not found at {MINUTES_MODEL_PATH}")
    return joblib.load(MINUTES_MODEL_PATH)


def load_historical_player_data() -> pd.DataFrame:
    """Load the most recent player game logs across seasons."""
    pattern = str(ROOT_DIR / "data" / "raw" / "player_logs" / "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No player log parquet files found.")
    
    # Load the latest 2 seasons for robust rolling estimates
    recent_files = files[-2:]
    dfs = [pd.read_parquet(f) for f in recent_files]
    df = pd.concat(dfs, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0.0)
    df["PLUS_MINUS"] = pd.to_numeric(df.get("PLUS_MINUS", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    return df


def build_team_rosters(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify the 15 primary active players for each team based on games played
    and recent involvement in the dataset.
    """
    rosters: Dict[str, List[Dict[str, Any]]] = {}

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

        top_15 = player_counts.head(15).to_dict(orient="records")
        rosters[abbrev] = top_15

    return rosters


def extract_player_minute_features(
    player_id: int,
    player_df: pd.DataFrame,
    feature_names: List[str],
    is_home: int = 1,
) -> Dict[str, float]:
    """
    Extract the 28 exact features required by the LightGBM Minutes Prediction Model:
    ['ROLL5_MIN_MEAN', 'ROLL5_MIN_STD', 'ROLL5_MIN_MAX', 'ROLL5_MIN_MIN',
     'ROLL10_MIN_MEAN', 'ROLL10_MIN_STD', 'ROLL10_MIN_MAX', 'ROLL10_MIN_MIN',
     'ROLL20_MIN_MEAN', 'ROLL20_MIN_STD', 'ROLL20_MIN_MAX', 'ROLL20_MIN_MIN',
     'SEASON_MIN_MEAN', 'CAREER_MIN_MEAN', 'MIN_TREND_5_10', 'MIN_TREND_10_20',
     'MIN_TREND_5_20', 'ROLL5_STARTER_PCT', 'ROLL10_STARTER_PCT', 'ROLL20_STARTER_PCT',
     'SEASON_STARTER_PCT', 'SEASON_GAMES_PLAYED', 'REST_DAYS', 'IS_B2B', 'IS_3IN4',
     'IS_HOME', 'OPP_ROLL10_PACE', 'OPP_ROLL10_DEF_RATING']
    """
    mins = player_df["MIN"].values
    n = len(mins)

    # Defaults
    feat: Dict[str, float] = {f: 0.0 for f in feature_names}

    if n == 0:
        feat["IS_HOME"] = float(is_home)
        feat["OPP_ROLL10_PACE"] = 100.0
        feat["OPP_ROLL10_DEF_RATING"] = 112.0
        return feat

    # Rolling windows
    def stats_for_window(w: int):
        sub = mins[-w:] if n >= w else mins
        mean_val = float(np.mean(sub))
        std_val = float(np.std(sub)) if len(sub) > 1 else 0.0
        max_val = float(np.max(sub))
        min_val = float(np.min(sub))
        starter_pct = float(np.mean(sub >= 24.0))
        return mean_val, std_val, max_val, min_val, starter_pct

    r5_mean, r5_std, r5_max, r5_min, r5_starters = stats_for_window(5)
    r10_mean, r10_std, r10_max, r10_min, r10_starters = stats_for_window(10)
    r20_mean, r20_std, r20_max, r20_min, r20_starters = stats_for_window(20)

    season_mean = float(np.mean(mins))
    career_mean = season_mean
    season_starters = float(np.mean(mins >= 24.0))

    feat["ROLL5_MIN_MEAN"] = r5_mean
    feat["ROLL5_MIN_STD"] = r5_std
    feat["ROLL5_MIN_MAX"] = r5_max
    feat["ROLL5_MIN_MIN"] = r5_min

    feat["ROLL10_MIN_MEAN"] = r10_mean
    feat["ROLL10_MIN_STD"] = r10_std
    feat["ROLL10_MIN_MAX"] = r10_max
    feat["ROLL10_MIN_MIN"] = r10_min

    feat["ROLL20_MIN_MEAN"] = r20_mean
    feat["ROLL20_MIN_STD"] = r20_std
    feat["ROLL20_MIN_MAX"] = r20_max
    feat["ROLL20_MIN_MIN"] = r20_min

    feat["SEASON_MIN_MEAN"] = season_mean
    feat["CAREER_MIN_MEAN"] = career_mean
    feat["MIN_TREND_5_10"] = r5_mean - r10_mean
    feat["MIN_TREND_10_20"] = r10_mean - r20_mean
    feat["MIN_TREND_5_20"] = r5_mean - r20_mean

    feat["ROLL5_STARTER_PCT"] = r5_starters
    feat["ROLL10_STARTER_PCT"] = r10_starters
    feat["ROLL20_STARTER_PCT"] = r20_starters
    feat["SEASON_STARTER_PCT"] = season_starters
    feat["SEASON_GAMES_PLAYED"] = float(n)

    feat["REST_DAYS"] = 2.0
    feat["IS_B2B"] = 0.0
    feat["IS_3IN4"] = 0.0
    feat["IS_HOME"] = float(is_home)
    feat["OPP_ROLL10_PACE"] = 100.0
    feat["OPP_ROLL10_DEF_RATING"] = 112.0

    return feat


def generate_all_team_predicted_minutes(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Predict minutes for each player in all 30 teams using the LightGBM Minutes Model.
    Normalizes the team minutes to strictly sum to 240.0 minutes.
    Caches results to disk.
    """
    if not force_refresh and CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)

    logger.info("Generating predicted minutes for all 30 teams...")
    model = load_minutes_model()
    feature_names = list(model.feature_name_)
    df = load_historical_player_data()
    rosters = build_team_rosters(df)

    team_rotations: Dict[str, Dict[str, Any]] = {}

    for abbrev, players in rosters.items():
        team_df = df[df["TEAM_ABBREVIATION"] == abbrev]
        predicted_players = []
        raw_pred_mins = []

        for p in players:
            p_id = p["PLAYER_ID"]
            p_name = p["PLAYER_NAME"]
            p_df = team_df[team_df["PLAYER_ID"] == p_id]

            feat_dict = extract_player_minute_features(p_id, p_df, feature_names)
            feat_df = pd.DataFrame([feat_dict])
            raw_min = float(model.predict(feat_df)[0])
            # Bound prediction to meaningful playing window
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

        # Sort by allocated minutes descending
        predicted_players.sort(key=lambda x: x["allocated_minutes"], reverse=True)

        # Categorize rotational roles
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
            "team_abbreviation": abbrev,
            "team_name": TEAM_TACTICAL_ACTIONS[abbrev]["team_name"],
            "total_allocated_minutes": round(sum(p["allocated_minutes"] for p in predicted_players), 2),
            "lineup": predicted_players,
        }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(team_rotations, f, indent=2)

    logger.info("Successfully generated and cached model-predicted minutes for all teams.")
    return team_rotations


if __name__ == "__main__":
    rotations = generate_all_team_predicted_minutes(force_refresh=True)
    print(f"Computed model-predicted minutes for {len(rotations)} teams.")
    bos = rotations["BOS"]["lineup"]
    print("\nBoston Celtics Model-Predicted 15-Man Lineup Rotation:")
    for p in bos:
        print(f"  [{p['role']:<11}] {p['player_name']:<24} Pred Min: {p['allocated_minutes']:>4.1f}m (Ratio: {p['minute_ratio']*100:>4.1f}%)")
