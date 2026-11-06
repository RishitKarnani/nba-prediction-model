"""
tests/test_minutes_features.py
──────────────────────────────
Tests for Expected Player Minutes feature engineering.

Covers:
  1. Rolling average minutes (5, 10, 20)
  2. Minutes consistency (std dev)
  3. Minutes trends (5-10, 10-20, 5-20)
  4. Season context (prior games played count, season average, starter proxy %)
  5. Strict zero data leakage verification (target game minutes never included)
  6. Snapshot utility for single-player inference
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from data.cleaner import clean_player_logs
from data.loader import get_player_logs
from features.minutes_features import (
    add_expected_minutes_features,
    get_player_minutes_snapshot,
)


@pytest.fixture(scope="module")
def sample_clean_players():
    raw = get_player_logs(seasons=["2023-24"])
    clean = clean_player_logs(raw)
    return clean


class TestExpectedMinutesFeatures:
    def test_columns_created(self, sample_clean_players):
        jokic = sample_clean_players[sample_clean_players["PLAYER_ID"] == 203999].copy()
        enriched = add_expected_minutes_features(jokic)

        expected_cols = [
            "ROLL5_MIN_MEAN",
            "ROLL10_MIN_MEAN",
            "ROLL20_MIN_MEAN",
            "ROLL5_MIN_STD",
            "ROLL10_MIN_STD",
            "MIN_TREND_5_10",
            "MIN_TREND_10_20",
            "MIN_TREND_5_20",
            "SEASON_MIN_MEAN",
            "SEASON_GAMES_PLAYED",
            "SEASON_STARTER_PCT",
            "ROLL5_STARTER_PCT",
            "ROLL10_STARTER_PCT",
        ]
        for col in expected_cols:
            assert col in enriched.columns, f"Missing column {col}"

    def test_minutes_trend_calculation(self, sample_clean_players):
        jokic = sample_clean_players[sample_clean_players["PLAYER_ID"] == 203999].copy()
        enriched = add_expected_minutes_features(jokic)

        # Check that trend equals the difference between the rolling means
        valid = enriched.dropna(subset=["ROLL5_MIN_MEAN", "ROLL10_MIN_MEAN", "MIN_TREND_5_10"])
        assert len(valid) > 0
        diff = (valid["ROLL5_MIN_MEAN"] - valid["ROLL10_MIN_MEAN"]) - valid["MIN_TREND_5_10"]
        assert (diff.abs() < 1e-6).all(), "MIN_TREND_5_10 does not match ROLL5 - ROLL10"

    def test_season_games_played_counter(self, sample_clean_players):
        jokic = sample_clean_players[sample_clean_players["PLAYER_ID"] == 203999].sort_values("GAME_DATE").reset_index(drop=True)
        enriched = add_expected_minutes_features(jokic)

        # First game of season must have SEASON_GAMES_PLAYED == 0
        assert enriched["SEASON_GAMES_PLAYED"].iloc[0] == 0

        # Each subsequent game increments by 1
        for i in range(1, min(10, len(enriched))):
            assert enriched["SEASON_GAMES_PLAYED"].iloc[i] == i

    def test_starter_pct_bounds(self, sample_clean_players):
        enriched = add_expected_minutes_features(sample_clean_players.head(200))
        valid_starter = enriched["SEASON_STARTER_PCT"].dropna()
        assert (valid_starter >= 0.0).all() and (valid_starter <= 1.0).all()

        valid_r5_starter = enriched["ROLL5_STARTER_PCT"].dropna()
        assert (valid_r5_starter >= 0.0).all() and (valid_r5_starter <= 1.0).all()


class TestMinutesDataLeakage:
    """CRITICAL: Verify Game N does NOT use Game N's actual minutes."""

    def test_rolling_minutes_excludes_current_game(self, sample_clean_players):
        jokic = sample_clean_players[sample_clean_players["PLAYER_ID"] == 203999].sort_values("GAME_DATE").reset_index(drop=True)
        enriched = add_expected_minutes_features(jokic)

        # Game at index 5 should have ROLL5_MIN_MEAN equal to mean of games 0..4
        for i in range(5, min(15, len(jokic))):
            expected = jokic["MIN"].iloc[i - 5 : i].mean()
            actual = enriched["ROLL5_MIN_MEAN"].iloc[i]
            assert abs(actual - expected) < 1e-4, (
                f"Data leakage at row {i}: expected {expected:.2f}, got {actual:.2f}"
            )

    def test_season_minutes_excludes_current_game(self, sample_clean_players):
        jokic = sample_clean_players[sample_clean_players["PLAYER_ID"] == 203999].sort_values("GAME_DATE").reset_index(drop=True)
        enriched = add_expected_minutes_features(jokic)

        # First game must have NaN for SEASON_MIN_MEAN
        assert np.isnan(enriched["SEASON_MIN_MEAN"].iloc[0])

        # Game 3's season mean must be mean of games 0..2
        expected = jokic["MIN"].iloc[:3].mean()
        actual = enriched["SEASON_MIN_MEAN"].iloc[3]
        assert abs(actual - expected) < 1e-4


class TestMinutesSnapshot:
    def test_snapshot_returns_valid_dict(self, sample_clean_players):
        snapshot = get_player_minutes_snapshot(sample_clean_players, player_id=203999)
        assert isinstance(snapshot, dict)
        assert snapshot["PLAYER_ID"] == 203999
        assert "ROLL5_MIN_MEAN" in snapshot
        assert "MIN_TREND_5_10" in snapshot
        assert "SEASON_GAMES_PLAYED" in snapshot
        assert snapshot["ROLL5_MIN_MEAN"] > 20.0  # Jokic plays > 20 mins


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

