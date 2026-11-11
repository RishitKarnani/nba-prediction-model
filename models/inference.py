"""
models/inference.py
───────────────────
End-to-end Version 1 Player Probability Inference Engine.

Connects:
  1. Preserved LightGBM Minutes Prediction Model
  2. Preserved 7 Champion LightGBM Statistical Models
  3. Chronological Out-of-Sample Residual Libraries (`ResidualIndex`)
  4. Validated Probability Estimation Method + Calibrator per statistic

Supports:
  - Querying a player vs. opponent for all 7 predicted statistics
  - Calculating Over/Under/Push probabilities across arbitrary lines
  - Reusing a single simulated outcome distribution across multiple lines
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from data.cleaner import clean_player_logs, clean_team_logs
from data.joiner import build_team_game_pairs
from data.loader import get_player_logs, get_team_logs, resolve_player_id, resolve_team_id
from features.minutes_features import add_expected_minutes_features
from features.player_features import (
    PROP_STATS,
    add_career_averages,
    add_home_away_splits,
    add_player_rolling_features,
    add_season_averages,
)
from features.schedule_features import add_rest_features
from models.calibration import ProbabilityCalibrator
from models.probability import (
    ConditionalConfig,
    LineProbabilityResult,
    OutcomeDistribution,
    ResidualIndex,
    simulate_player_outcomes,
)
from models.residuals import TARGETS, canonical_stat_name, load_oos_predictions

logger = logging.getLogger(__name__)


class PlayerProbabilityEngine:
    """
    Production inference interface for Version 1 NBA Player Performance & Probability Engine.
    Strictly estimates player performance and threshold probabilities — no odds or EV logic.
    """

    def __init__(
        self,
        artefacts_dir: Optional[Path] = None,
        residual_cutoff_date: Optional[str | pd.Timestamp] = None,
        residual_seasons: Optional[List[str]] = None,
    ):
        self.artefacts_dir = artefacts_dir or settings.artefacts_dir
        self.prob_dir = self.artefacts_dir / "probability"

        # 1. Load preserved Minutes Prediction Model
        min_path = self.artefacts_dir / "models" / "minutes" / "minutes_model.joblib"
        if not min_path.exists():
            raise FileNotFoundError(f"Minutes model not found at {min_path}")
        self.minutes_model = joblib.load(min_path)
        self.minutes_features: List[str] = list(self.minutes_model.feature_name_)

        # 2. Load 7 preserved Champion Statistical Models
        self.stat_models: Dict[str, Any] = {}
        self.stat_features: Dict[str, List[str]] = {}
        for stat in TARGETS:
            m_path = self.artefacts_dir / "champions" / f"champion_{stat.lower()}.joblib"
            if not m_path.exists():
                raise FileNotFoundError(f"Champion model not found at {m_path}")
            model = joblib.load(m_path)
            self.stat_models[stat] = model
            self.stat_features[stat] = list(model.feature_name_)

        # 3. Load probability engine config (selected method & conditional config per stat)
        cfg_path = self.prob_dir / "engine_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.engine_config = json.load(f)
        else:
            self.engine_config = {
                "selected_methods": {s: "conditional_bootstrap" for s in TARGETS},
                "conditional_config": {},
            }

        cond_kwargs = self.engine_config.get("conditional_config", {})
        self.conditional_config = ConditionalConfig(**cond_kwargs) if cond_kwargs else ConditionalConfig()

        # 4. Load calibrators if persisted
        cal_path = self.prob_dir / "calibrators.joblib"
        if cal_path.exists():
            self.calibrators: Dict[str, ProbabilityCalibrator] = joblib.load(cal_path)
        else:
            self.calibrators = {s: ProbabilityCalibrator(stat=s, method="none", status="uncalibrated") for s in TARGETS}

        # 5. Build ResidualIndex per stat strictly from out-of-sample history
        self.residual_indices: Dict[str, ResidualIndex] = {}
        for stat in TARGETS:
            oos_df = load_oos_predictions(
                stat=stat,
                seasons=residual_seasons,
                max_date=residual_cutoff_date,
            )
            self.residual_indices[stat] = ResidualIndex.from_dataframe(
                oos_df,
                stat=stat,
                config=self.conditional_config,
            )

        # Lazily cached raw logs for live matchup construction
        self._clean_players_cache: Optional[pd.DataFrame] = None
        self._team_logs_cache: Optional[pd.DataFrame] = None

    def _get_clean_players(self) -> pd.DataFrame:
        if self._clean_players_cache is None:
            self._clean_players_cache = clean_player_logs(get_player_logs())
        return self._clean_players_cache

    def _get_clean_teams(self) -> pd.DataFrame:
        if self._team_logs_cache is None:
            self._team_logs_cache = clean_team_logs(get_team_logs())
        return self._team_logs_cache

    def build_pregame_feature_row(
        self,
        player: str | int,
        opponent: str | int,
        is_home: int = 1,
        game_date: Optional[str | pd.Timestamp] = None,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Construct an exact, leakage-free pre-game feature vector for an upcoming
        matchup by appending a synthetic game row strictly after the player's
        completed history before `game_date`.
        """
        if isinstance(player, int):
            player_id = player
        else:
            player_id = resolve_player_id(player)
            if player_id is None:
                raise ValueError(f"Could not resolve player '{player}'.")

        if isinstance(opponent, int):
            opp_team_id = opponent
        else:
            opp_team_id = resolve_team_id(opponent)
            if opp_team_id is None:
                raise ValueError(f"Could not resolve opponent team '{opponent}'.")

        clean_p = self._get_clean_players()
        clean_t = self._get_clean_teams()

        p_hist = clean_p[clean_p["PLAYER_ID"] == player_id].copy()
        if p_hist.empty:
            raise ValueError(f"No historical logs found for PLAYER_ID={player_id}.")

        p_hist = p_hist.sort_values("GAME_DATE").reset_index(drop=True)
        opp_rows = clean_t[clean_t["TEAM_ID"] == opp_team_id].sort_values("GAME_DATE")
        if opp_rows.empty:
            raise ValueError(f"No historical logs found for OPP_TEAM_ID={opp_team_id}.")
        opp_abbrev = str(opp_rows.iloc[-1]["TEAM_ABBREVIATION"])
        opp_team_name = str(opp_rows.iloc[-1].get("TEAM_NAME", opp_abbrev))

        if game_date is None:
            target_dt = p_hist["GAME_DATE"].max() + pd.Timedelta(days=2)
        else:
            target_dt = pd.to_datetime(game_date)
            # Strict chronological cutoff so backtest/hypothetical queries never peek ahead
            p_hist = p_hist[p_hist["GAME_DATE"] < target_dt].copy()
            if p_hist.empty:
                raise ValueError(f"No player games prior to {target_dt} for PLAYER_ID={player_id}.")

        last_row = p_hist.iloc[-1].copy()
        latest_season = str(last_row["SEASON"])
        player_name = str(last_row["PLAYER_NAME"])
        team_abbrev = str(last_row["TEAM_ABBREVIATION"])

        # Synthetic upcoming game row (all target stats = NaN so shift(1) includes all past games cleanly)
        upcoming_row = last_row.copy()
        upcoming_row["GAME_ID"] = "UPCOMING_QUERY"
        upcoming_row["GAME_DATE"] = target_dt
        upcoming_row["SEASON"] = latest_season
        upcoming_row["IS_HOME"] = int(is_home)
        upcoming_row["OPPONENT_ABBREV"] = opp_abbrev
        upcoming_row["MATCHUP"] = f"{team_abbrev} {'vs.' if is_home else '@'} {opp_abbrev}"
        for col in ["MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
                    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]:
            if col in upcoming_row.index:
                upcoming_row[col] = np.nan

        p_with_target = pd.concat([p_hist, pd.DataFrame([upcoming_row])], ignore_index=True)

        # Compute exact pre-game rolling / season / career / schedule features via existing pipeline functions
        p_feat = add_player_rolling_features(p_with_target)
        p_feat = add_expected_minutes_features(p_feat)
        p_feat = add_season_averages(p_feat, stats=PROP_STATS)
        p_feat = add_career_averages(p_feat, stats=PROP_STATS)
        p_feat = add_home_away_splits(p_feat, stats=["PTS", "REB", "AST"])
        p_feat = add_rest_features(p_feat, id_col="PLAYER_ID")

        row = p_feat[p_feat["GAME_ID"] == "UPCOMING_QUERY"].iloc[-1].copy()

        # Compute opponent pre-game rolling ratings & pace strictly prior to target_dt
        opp_t_hist = clean_t[clean_t["GAME_DATE"] < target_dt].copy()
        paired = build_team_game_pairs(opp_t_hist)
        opp_paired = paired[paired["TEAM_ID"] == opp_team_id].sort_values("GAME_DATE")

        for w in settings.ROLLING_WINDOWS:
            recent_t = opp_paired.tail(w)
            for metric in ["DEF_RATING", "NET_RATING", "PACE"]:
                col_name = f"OPP_ROLL{w}_{metric}"
                row[col_name] = float(recent_t[metric].mean()) if not recent_t.empty and metric in recent_t.columns else np.nan

        # Opponent conceded PTS over last W games strictly prior to target_dt
        opp_def_players = clean_p[(clean_p["OPPONENT_ABBREV"] == opp_abbrev) & (clean_p["GAME_DATE"] < target_dt) & (clean_p["MIN"] >= settings.MIN_MINUTES_THRESHOLD)]
        if not opp_def_players.empty:
            game_conceded = opp_def_players.groupby(["GAME_ID", "GAME_DATE"])["PTS"].sum().reset_index().sort_values("GAME_DATE")
            for w in settings.ROLLING_WINDOWS:
                row[f"OPP_CONCEDED_PTS_ROLL{w}"] = float(game_conceded["PTS"].tail(w).mean()) if not game_conceded.empty else np.nan
        else:
            for w in settings.ROLLING_WINDOWS:
                row[f"OPP_CONCEDED_PTS_ROLL{w}"] = np.nan

        row["OPP_TEAM_ID"] = opp_team_id

        meta = {
            "player_id": player_id,
            "player_name": player_name,
            "team_abbreviation": team_abbrev,
            "opponent_team_id": opp_team_id,
            "opponent_abbrev": opp_abbrev,
            "opponent_name": opp_team_name,
            "is_home": int(is_home),
            "game_date": target_dt.strftime("%Y-%m-%d"),
            "n_prior_games": int(len(p_hist)),
        }
        return row, meta

    def predict_feature_row(
        self,
        feature_row: pd.Series,
        meta: Optional[Dict[str, Any]] = None,
        lines_by_stat: Optional[Dict[str, List[float]]] = None,
        method_override: Optional[str] = None,
        n_simulations: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Given a single pre-game `feature_row` Series:
          1. Predict expected minutes using preserved Minutes Model
          2. Predict all 7 statistics using preserved Champion LightGBM models
          3. Simulate outcome distributions & evaluate any user-supplied lines
        """
        # 1. Predict minutes
        min_df = pd.DataFrame([{c: feature_row.get(c, np.nan) for c in self.minutes_features}])
        pred_min = float(np.clip(self.minutes_model.predict(min_df)[0], 5.0, 48.0))

        # Attach predicted_minutes for v2 statistical models
        enriched_row = feature_row.copy()
        enriched_row["predicted_minutes"] = pred_min

        # Determine starter flag for conditional bootstrap context
        starter_pct = enriched_row.get("ROLL5_STARTER_PCT", np.nan)
        roll5_min = enriched_row.get("ROLL5_MIN_MEAN", np.nan)
        if pd.notna(starter_pct):
            is_starter: Optional[bool] = bool(starter_pct >= 0.5)
        elif pd.notna(roll5_min):
            is_starter = bool(roll5_min >= 24.0)
        else:
            is_starter = None

        n_sim = n_simulations or self.conditional_config.n_simulations

        predicted_stats: Dict[str, float] = {}
        distributions: Dict[str, OutcomeDistribution] = {}
        line_results: Dict[str, List[Dict[str, Any]]] = {}

        for stat in TARGETS:
            req_cols = self.stat_features[stat]
            stat_df = pd.DataFrame([{c: enriched_row.get(c, np.nan) for c in req_cols}])
            pred_val = float(max(0.0, self.stat_models[stat].predict(stat_df)[0]))
            predicted_stats[stat] = round(pred_val, 2)

            chosen_method = method_override or self.engine_config.get("selected_methods", {}).get(stat, "conditional_bootstrap")
            dist = simulate_player_outcomes(
                stat=stat,
                predicted_value=pred_val,
                predicted_minutes=pred_min,
                residual_index=self.residual_indices[stat],
                method=chosen_method,
                is_starter=is_starter,
                n_simulations=n_sim,
                seed=seed,
                calibrator=self.calibrators.get(stat),
            )
            distributions[stat] = dist

            if lines_by_stat and stat in lines_by_stat:
                res_list = dist.probabilities_for_lines(lines_by_stat[stat])
                line_results[stat] = [r.to_dict() for r in res_list]

        return {
            "metadata": meta or {},
            "predicted_minutes": round(pred_min, 2),
            "is_starter_context": is_starter,
            "predicted_stats": predicted_stats,
            "distributions": distributions,
            "line_probabilities": line_results,
        }

    def predict_player(
        self,
        player: str | int,
        opponent: str | int,
        is_home: int = 1,
        game_date: Optional[str | pd.Timestamp] = None,
        lines_by_stat: Optional[Dict[str, List[float]]] = None,
        method_override: Optional[str] = None,
        n_simulations: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        End-to-end query for any NBA player and opponent matchup.

        Example:
            engine.predict_player(
                player="Stephen Curry",
                opponent="Los Angeles Lakers",
                lines_by_stat={"PTS": [20.5, 25.5, 30.5], "3PM": [3.5, 4.5]},
            )
        """
        # Normalize keys in lines_by_stat cleanly
        norm_lines: Optional[Dict[str, List[float]]] = None
        if lines_by_stat:
            norm_lines = {canonical_stat_name(k): v for k, v in lines_by_stat.items()}

        row, meta = self.build_pregame_feature_row(
            player=player,
            opponent=opponent,
            is_home=is_home,
            game_date=game_date,
        )
        return self.predict_feature_row(
            feature_row=row,
            meta=meta,
            lines_by_stat=norm_lines,
            method_override=method_override,
            n_simulations=n_simulations,
            seed=seed,
        )

