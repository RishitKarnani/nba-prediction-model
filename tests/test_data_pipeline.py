"""
tests/test_data_pipeline.py
────────────────────────────
Tests for data loading, cleaning, joining, and feature engineering.

Covers:
  1. Loader: files load, shapes are correct, date range is as expected
  2. Cleaner: null handling, derived columns, IS_HOME accuracy
  3. Joiner: paired rows are correct, DEF_RATING is computed
  4. Schedule features: REST_DAYS, IS_B2B correctness
  5. DATA LEAKAGE CHECK: rolling features must not include current game
  6. Player rolling features: shift(1) verification
"""

from __future__ import annotations

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from data.cleaner import clean_player_logs, clean_team_logs
from data.joiner import build_team_game_pairs
from data.loader import get_player_logs, get_team_logs, resolve_player_id, resolve_team_id
from features.player_features import (
    add_player_rolling_features,
    add_season_averages,
    get_player_game_history,
)
from features.schedule_features import add_rest_features
from features.team_features import add_team_rolling_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_players():
    return get_player_logs()

@pytest.fixture(scope="module")
def raw_teams():
    return get_team_logs()

@pytest.fixture(scope="module")
def clean_players(raw_players):
    return clean_player_logs(raw_players)

@pytest.fixture(scope="module")
def clean_teams(raw_teams):
    return clean_team_logs(raw_teams)

@pytest.fixture(scope="module")
def paired_teams(clean_teams):
    return build_team_game_pairs(clean_teams)


# ── 1. Loader tests ───────────────────────────────────────────────────────────

class TestLoader:
    def test_player_logs_shape(self, raw_players):
        assert len(raw_players) > 300_000, "Expected > 300K player log rows"
        assert "PLAYER_ID" in raw_players.columns
        assert "GAME_ID" in raw_players.columns
        assert "PTS" in raw_players.columns

    def test_team_logs_shape(self, raw_teams):
        assert len(raw_teams) > 29_000, "Expected > 29K team log rows"
        assert "TEAM_ID" in raw_teams.columns
        assert "PTS" in raw_teams.columns

    def test_player_logs_seasons(self, raw_players):
        seasons = raw_players["SEASON"].unique()
        assert "2023-24" in seasons
        assert "2026-27" in seasons
        assert len(seasons) == 13

    def test_game_date_parsed(self, raw_players):
        assert pd.api.types.is_datetime64_any_dtype(raw_players["GAME_DATE"])

    def test_resolve_player_id_exact(self):
        pid = resolve_player_id("LeBron James")
        assert pid == 2544, f"Expected 2544 for LeBron James, got {pid}"

    def test_resolve_player_id_partial(self):
        pid = resolve_player_id("Jokic")
        assert pid == 203999, f"Expected 203999 for Jokic, got {pid}"

    def test_resolve_player_id_not_found(self):
        pid = resolve_player_id("XXXXXXXX_not_a_real_player_12345")
        assert pid is None

    def test_resolve_team_id_abbrev(self):
        tid = resolve_team_id("LAL")
        assert tid is not None
        assert isinstance(tid, int)

    def test_resolve_team_id_name(self):
        tid = resolve_team_id("Nuggets")
        assert tid == 1610612743


# ── 2. Cleaner tests ──────────────────────────────────────────────────────────

class TestCleaner:
    def test_no_nan_fg_pct(self, clean_players):
        assert clean_players["FG_PCT"].isna().sum() == 0

    def test_no_nan_fg3_pct(self, clean_players):
        assert clean_players["FG3_PCT"].isna().sum() == 0

    def test_no_nan_ft_pct(self, clean_players):
        assert clean_players["FT_PCT"].isna().sum() == 0

    def test_is_home_derived(self, clean_players):
        home_rows = clean_players[clean_players["MATCHUP"].str.contains("vs.", na=False)]
        assert (home_rows["IS_HOME"] == 1).all(), "Home rows should have IS_HOME=1"

        away_rows = clean_players[clean_players["MATCHUP"].str.contains("@ ", na=False)]
        assert (away_rows["IS_HOME"] == 0).all(), "Away rows should have IS_HOME=0"

    def test_efg_pct_range(self, clean_players):
        valid = clean_players[clean_players["FGA"] > 0]["EFG_PCT"]
        assert (valid >= 0).all() and (valid <= 1.5).all(), "eFG% should be in [0, 1.5]"

    def test_opponent_abbrev_not_own(self, clean_players):
        # Opponent abbreviation should never match the player's own team
        has_opp = clean_players[clean_players["OPPONENT_ABBREV"] != ""]
        bad = has_opp[has_opp["OPPONENT_ABBREV"] == has_opp["TEAM_ABBREVIATION"]]
        assert len(bad) == 0, f"{len(bad)} rows where OPPONENT_ABBREV == own team"

    def test_team_possessions_positive(self, clean_teams):
        assert (clean_teams["POSSESSIONS"] > 0).all()

    def test_team_off_rating_range(self, clean_teams):
        # NBA offensive ratings typically 90-130 per 100 possessions.
        # Allow lower bound of 50 for unusual game situations (OT, low possession counts).
        assert (clean_teams["OFF_RATING"] > 50).all(), "OFF_RATING should be > 50 for all games"
        assert (clean_teams["OFF_RATING"] < 180).all(), "OFF_RATING should be < 180 for all games"


# ── 3. Joiner tests ───────────────────────────────────────────────────────────

class TestJoiner:
    def test_paired_shape(self, paired_teams):
        # Each game has exactly 2 teams, so pairs = 2 * n_games
        assert len(paired_teams) == 29054, f"Expected 29054 paired rows, got {len(paired_teams)}"

    def test_no_self_join(self, paired_teams):
        assert (paired_teams["TEAM_ID"] != paired_teams["OPP_TEAM_ID"]).all()

    def test_opp_pts_present(self, paired_teams):
        assert "OPP_PTS" in paired_teams.columns
        assert paired_teams["OPP_PTS"].notna().all()

    def test_def_rating_computed(self, paired_teams):
        assert "DEF_RATING" in paired_teams.columns
        assert (paired_teams["DEF_RATING"] > 50).all()
        assert (paired_teams["DEF_RATING"] < 200).all()

    def test_net_rating_equals_off_minus_def(self, paired_teams):
        diff = (paired_teams["NET_RATING"] - (paired_teams["OFF_RATING"] - paired_teams["DEF_RATING"])).abs()
        assert diff.max() < 0.001, "NET_RATING should equal OFF_RATING - DEF_RATING"

    def test_win_from_wl(self, paired_teams):
        w_rows = paired_teams[paired_teams["WL"] == "W"]
        l_rows = paired_teams[paired_teams["WL"] == "L"]
        assert (w_rows["WIN"] == 1).all()
        assert (l_rows["WIN"] == 0).all()


# ── 4. Schedule features tests ────────────────────────────────────────────────

class TestScheduleFeatures:
    def test_rest_days_non_negative(self, clean_teams):
        df = add_rest_features(clean_teams, id_col="TEAM_ID")
        rest = df["REST_DAYS"].dropna()
        assert (rest >= 1).all(), "REST_DAYS should be >= 1 (minimum 1 day between games)"

    def test_b2b_when_rest_is_1(self, clean_teams):
        df = add_rest_features(clean_teams, id_col="TEAM_ID")
        b2b_rows = df[df["IS_B2B"] == 1]
        assert (b2b_rows["REST_DAYS"] == 1).all(), "IS_B2B=1 only when REST_DAYS=1"

    def test_first_game_has_nan_rest(self, clean_teams):
        df = add_rest_features(clean_teams, id_col="TEAM_ID")
        first_games = df.groupby("TEAM_ID").head(1)
        assert first_games["REST_DAYS"].isna().all(), "First game per team should have NaN REST_DAYS"


# ── 5. DATA LEAKAGE CHECK ─────────────────────────────────────────────────────

class TestNoDataLeakage:
    """
    CRITICAL: Verify that rolling features computed for game N do NOT include
    game N's stats. This is the most important correctness test.
    """

    def test_player_rolling_excludes_current_game(self, clean_players):
        # Take one player with many games: Nikola Jokic (203999)
        player_id = 203999
        pdf = clean_players[clean_players["PLAYER_ID"] == player_id].copy()
        pdf = pdf[pdf["MIN"] >= 5].sort_values("GAME_DATE").reset_index(drop=True)

        if len(pdf) < 10:
            pytest.skip("Not enough games for this player")

        enriched = add_player_rolling_features(pdf)

        # For game at index 5, ROLL5_PTS_MEAN should equal mean of games 0-4
        if "ROLL5_PTS_MEAN" not in enriched.columns:
            pytest.skip("ROLL5_PTS_MEAN not computed")

        for i in range(5, min(15, len(pdf))):
            expected = pdf["PTS"].iloc[max(0, i - 5):i].mean()
            actual = enriched["ROLL5_PTS_MEAN"].iloc[i]
            if not np.isnan(actual):
                assert abs(actual - expected) < 0.01, (
                    f"Leakage detected at row {i}: expected {expected:.2f}, got {actual:.2f}"
                )

    def test_team_rolling_excludes_current_game(self, paired_teams):
        # Take Denver Nuggets
        team_id = 1610612743
        tdf = paired_teams[paired_teams["TEAM_ID"] == team_id].sort_values("GAME_DATE").reset_index(drop=True)

        if len(tdf) < 10:
            pytest.skip("Not enough games for this team")

        enriched = add_team_rolling_features(tdf)

        if "ROLL5_PTS" not in enriched.columns:
            pytest.skip("ROLL5_PTS not computed")

        for i in range(5, min(15, len(tdf))):
            expected = tdf["PTS"].iloc[max(0, i - 5):i].mean()
            actual = enriched["ROLL5_PTS"].iloc[i]
            if not np.isnan(actual):
                assert abs(actual - expected) < 0.01, (
                    f"Team leakage detected at row {i}: expected {expected:.2f}, got {actual:.2f}"
                )


# ── 6. Player rolling feature tests ──────────────────────────────────────────

class TestPlayerRollingFeatures:
    def test_rolling_columns_created(self, clean_players):
        sample = clean_players[clean_players["PLAYER_ID"] == 203999].head(30).copy()
        enriched = add_player_rolling_features(sample)

        for w in [5, 10, 20]:
            assert f"ROLL{w}_PTS_MEAN" in enriched.columns
            assert f"ROLL{w}_PTS_STD" in enriched.columns

    def test_season_avg_excludes_current_game(self, clean_players):
        sample = clean_players[clean_players["PLAYER_ID"] == 203999].copy()
        sample = sample[sample["SEASON"] == "2023-24"].sort_values("GAME_DATE").reset_index(drop=True)

        if len(sample) < 5:
            pytest.skip("Not enough 2023-24 games")

        enriched = add_season_averages(sample, stats=["PTS"])

        # Row 3's season avg should be mean of rows 0–2
        i = 3
        expected = sample["PTS"].iloc[:i].mean()
        actual = enriched["SEASON_PTS_MEAN"].iloc[i]
        if not np.isnan(actual):
            assert abs(actual - expected) < 0.01

    def test_game_history_returns_correct_count(self, clean_players):
        history = get_player_game_history(clean_players, player_id=203999, stat="PTS", n_games=10)
        assert len(history) <= 10
        assert len(history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
