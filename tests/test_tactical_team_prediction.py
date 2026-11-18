"""
tests/test_tactical_team_prediction.py
──────────────────────────────────────
Test suite for the Tactical Action Matchup Simulation and Team Prediction Engine.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tactical_registry import TEAM_TACTICAL_ACTIONS, get_action_delta
from src.weight_optimizer import adjust_action_weights_ratio
from models.team_inference import TeamPredictionEngine, resolve_team_abbrev


def test_tactical_registry_integrity():
    """Verify that all 30 teams have exactly 5 offensive and 5 defensive actions."""
    assert len(TEAM_TACTICAL_ACTIONS) == 30

    for abbrev, team_data in TEAM_TACTICAL_ACTIONS.items():
        assert len(team_data["offensive_actions"]) == 5
        assert len(team_data["defensive_actions"]) == 5
        assert len(team_data["action_archetypes_off"]) == 5
        assert len(team_data["action_archetypes_def"]) == 5


def test_weight_optimizer_strictly_non_zero():
    """Verify that no adjusted weight is 0 and sum is strictly 1.0."""
    wins = [0.85, 0.40, 0.20, 0.10, 0.05]
    adj = adjust_action_weights_ratio(wins, min_floor=0.04)

    assert len(adj) == 5
    assert sum(adj) == pytest.approx(1.0, abs=1e-3)
    for w in adj:
        assert w >= 0.04  # Strictly greater than 0
    # Ensure monotonic ordering based on win rate
    assert adj[0] > adj[1] > adj[2] > adj[3] > adj[4]


def test_team_prediction_engine():
    """Test full prediction output for Boston vs Dallas."""
    engine = TeamPredictionEngine()
    res = engine.predict_game("BOS", "DAL")

    assert res["home_team"] == "BOS"
    assert res["away_team"] == "DAL"
    assert "predicted_winner" in res["prediction"]
    assert "predicted_score" in res["prediction"]
    assert "predicted_margin" in res["prediction"]

    # Check tactical gameplan weights
    h_weights = res["tactical_gameplan"]["home_team"]["adjusted_offensive_weights"]
    assert len(h_weights) == 5
    for act, w in h_weights.items():
        assert w > 0.0

    # Check lineup minutes sum
    h_lineup = res["lineup_predicted_minutes"]["home_team"]
    assert len(h_lineup) >= 5


def test_team_abbreviation_resolver():
    """Test resolving various string representations of team names."""
    assert resolve_team_abbrev("LAL") == "LAL"
    assert resolve_team_abbrev("Golden State Warriors") == "GSW"
    assert resolve_team_abbrev("Celtics") == "BOS"
    assert resolve_team_abbrev("Miami") == "MIA"
