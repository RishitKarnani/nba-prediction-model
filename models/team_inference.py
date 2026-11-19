"""
models/team_inference.py
────────────────────────
End-to-End Team Game Outcome & Scoring Margin Prediction Engine.

Integrates:
  1. 15-Man Active Rosters with Model-Predicted Playing Time (LightGBM).
  2. Tactical Action Synergy Matrix (5 Offensive Actions vs 5 Defensive Coverages).
  3. Non-Zero Weight Rebalancing based on Opponent-Specific Win Expectation.
  4. Winner, Win Probability, Projected Scores, Point Spread Margin, and Tactical Gameplan.
  5. [NEW] Team Form Factor (50% Last-5 / 30% Last-10 / 20% Season).
  6. [NEW] Injury Engine — minute/scoring redistribution for OUT players.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from data.loader import resolve_team_id
from src.matchup_simulator import (
    SIMULATION_RESULTS_PATH,
    calculate_lineup_impact,
    calculate_tactical_action_expectations,
    load_team_baseline_ratings,
    run_full_league_simulation,
    simulate_game,
)
from src.roster_minutes import CACHE_PATH as ROSTER_CACHE_PATH, generate_all_team_predicted_minutes
from src.tactical_registry import TEAM_TACTICAL_ACTIONS
from src.weight_optimizer import optimize_team_tactical_weights
from src.form_factor import compute_team_form, get_form_adjusted_baselines
from src.injury_engine import apply_injuries, format_injury_report

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent


def resolve_team_abbrev(team_str: str) -> str:
    """
    Resolve a team name, city, or abbreviation to the canonical 3-letter abbreviation.
    """
    s = team_str.strip().upper()
    if s in TEAM_TACTICAL_ACTIONS:
        return s

    # Search by full name or city substring
    for abbrev, data in TEAM_TACTICAL_ACTIONS.items():
        name = data["team_name"].upper()
        if s in name or name in s:
            return abbrev

    raise ValueError(f"Could not resolve team identifier '{team_str}'. Available: {list(TEAM_TACTICAL_ACTIONS.keys())}")


class TeamPredictionEngine:
    """
    Inference engine predicting NBA game winners, point margins, and tactical matchups.

    Parameters
    ----------
    auto_load_simulation : bool
        Load pre-computed 8,700-game simulation results from cache.
    apply_form : bool
        If True (default), use form-weighted baselines (50/30/20) instead of raw season averages.
    """

    def __init__(self, auto_load_simulation: bool = True, apply_form: bool = True):
        self.apply_form = apply_form

        # Load simulation results
        self.sim_results = None
        if auto_load_simulation and SIMULATION_RESULTS_PATH.exists():
            with open(SIMULATION_RESULTS_PATH, "r") as f:
                self.sim_results = json.load(f)
        else:
            self.sim_results = run_full_league_simulation(sims_per_matchup=10)

        # Load rosters
        if ROSTER_CACHE_PATH.exists():
            with open(ROSTER_CACHE_PATH, "r") as f:
                self.rosters = json.load(f)
        else:
            self.rosters = generate_all_team_predicted_minutes()

        # Load baselines (and optionally apply form adjustment)
        raw_baselines = load_team_baseline_ratings()
        if self.apply_form:
            try:
                self.baselines = get_form_adjusted_baselines(raw_baselines)
                self.form_data = compute_team_form()
                logger.info("Form-adjusted baselines loaded successfully.")
            except Exception as e:
                logger.warning(f"Form factor failed ({e}), falling back to raw baselines.")
                self.baselines = raw_baselines
                self.form_data = {}
        else:
            self.baselines = raw_baselines
            self.form_data = {}

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        n_simulations: int = 500,
        injuries_home: Optional[List[str]] = None,
        injuries_away: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict game outcome, winning margin, final score, and tactical gameplans
        using a 500-draw Monte Carlo simulation average with deterministic seed.

        Parameters
        ----------
        home_team : str
            Home team abbreviation or name.
        away_team : str
            Away team abbreviation or name.
        n_simulations : int
            Number of Monte Carlo simulation draws (default 500 for high resolution).
        injuries_home : list of str, optional
            List of home team player names who are OUT.
        injuries_away : list of str, optional
            List of away team player names who are OUT.
        """
        home_abbrev = resolve_team_abbrev(home_team)
        away_abbrev = resolve_team_abbrev(away_team)

        if home_abbrev == away_abbrev:
            raise ValueError("Home and Away teams cannot be the same.")

        fixture_key = f"{away_abbrev}_at_{home_abbrev}"

        # ── Injury processing ─────────────────────────────────────────────────
        injuries_home = injuries_home or []
        injuries_away = injuries_away or []

        h_lineup_raw = self.rosters[home_abbrev]["lineup"]
        a_lineup_raw = self.rosters[away_abbrev]["lineup"]

        h_injury_ortg_adj = 0.0
        a_injury_ortg_adj = 0.0
        h_injury_summary = {"injuries": [], "pts_lost": 0.0, "ortg_adjustment": 0.0}
        a_injury_summary = {"injuries": [], "pts_lost": 0.0, "ortg_adjustment": 0.0}

        if injuries_home:
            h_lineup_raw, h_injury_ortg_adj, h_injury_summary = apply_injuries(h_lineup_raw, injuries_home)
        if injuries_away:
            a_lineup_raw, a_injury_ortg_adj, a_injury_summary = apply_injuries(a_lineup_raw, injuries_away)

        # ── Lineup impact (tactical quality score) ────────────────────────────
        h_lineup_adj = calculate_lineup_impact(h_lineup_raw)
        a_lineup_adj = calculate_lineup_impact(a_lineup_raw)

        # ── Build per-game baselines (may be form-adjusted) ───────────────────
        # Apply injury ortg adjustments on top of form-adjusted baselines
        h_baselines = dict(self.baselines[home_abbrev])
        a_baselines = dict(self.baselines[away_abbrev])
        h_baselines["base_ortg"] = h_baselines["base_ortg"] + h_injury_ortg_adj
        a_baselines["base_ortg"] = a_baselines["base_ortg"] + a_injury_ortg_adj

        # ── Tactical weights ──────────────────────────────────────────────────
        h_net = h_baselines["base_ortg"] - h_baselines["base_drtg"] + h_lineup_adj + 2.8
        a_net = a_baselines["base_ortg"] - a_baselines["base_drtg"] + a_lineup_adj
        expected_base_margin = h_net - a_net

        h_off_wins, h_def_wins = calculate_tactical_action_expectations(
            home_abbrev, away_abbrev, baseline_margin=expected_base_margin, is_home_a=True
        )
        a_off_wins, a_def_wins = calculate_tactical_action_expectations(
            away_abbrev, home_abbrev, baseline_margin=-expected_base_margin, is_home_a=False
        )

        h_tactics = TEAM_TACTICAL_ACTIONS[home_abbrev]
        a_tactics = TEAM_TACTICAL_ACTIONS[away_abbrev]

        h_opt = optimize_team_tactical_weights(
            h_off_wins, h_def_wins, h_tactics["offensive_actions"], h_tactics["defensive_actions"]
        )
        a_opt = optimize_team_tactical_weights(
            a_off_wins, a_def_wins, a_tactics["offensive_actions"], a_tactics["defensive_actions"]
        )

        # ── Monte Carlo simulation (deterministic per fixture) ───────────────
        seed_val = abs(hash(fixture_key)) % (2**31)
        np.random.seed(seed_val)

        home_wins = 0
        home_pts_list = []
        away_pts_list = []
        margins = []

        for _ in range(n_simulations):
            res = simulate_game(
                home_team=home_abbrev,
                away_team=away_abbrev,
                home_baselines=h_baselines,
                away_baselines=a_baselines,
                home_lineup_adj=h_lineup_adj,
                away_lineup_adj=a_lineup_adj,
                home_off_weights=h_opt["raw_off_weights_list"],
                home_def_weights=h_opt["raw_def_weights_list"],
                away_off_weights=a_opt["raw_off_weights_list"],
                away_def_weights=a_opt["raw_def_weights_list"],
            )
            home_pts_list.append(res["pts_home"])
            away_pts_list.append(res["pts_away"])
            margins.append(res["margin"])
            if res["winner"] == home_abbrev:
                home_wins += 1

        home_win_prob = round(home_wins / n_simulations, 3)
        avg_margin = round(float(sum(margins) / len(margins)), 1)
        pred_winner = home_abbrev if avg_margin >= 0 else away_abbrev

        # ── Build structured result ───────────────────────────────────────────
        winner_name = (
            h_tactics["team_name"] if pred_winner == home_abbrev else a_tactics["team_name"]
        )
        margin_val = abs(avg_margin)
        margin_str = f"+{margin_val:.1f} ({pred_winner})"

        # Form metadata for display
        h_form = self.form_data.get(home_abbrev, {})
        a_form = self.form_data.get(away_abbrev, {})

        return {
            "fixture": f"{away_abbrev}_at_{home_abbrev}",
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_team_name": h_tactics["team_name"],
            "away_team_name": a_tactics["team_name"],
            "form_applied": self.apply_form,
            "form": {
                home_abbrev: h_form,
                away_abbrev: a_form,
            },
            "injuries": {
                home_abbrev: h_injury_summary,
                away_abbrev: a_injury_summary,
            },
            "prediction": {
                "predicted_winner":    winner_name,
                "winner_abbreviation": pred_winner,
                "home_win_probability": home_win_prob,
                "away_win_probability": round(1.0 - home_win_prob, 3),
                "predicted_score": {
                    home_abbrev: round(float(sum(home_pts_list) / len(home_pts_list)), 1),
                    away_abbrev: round(float(sum(away_pts_list) / len(away_pts_list)), 1),
                },
                "predicted_margin": margin_str,
                "implied_spread": f"{pred_winner} -{margin_val:.1f}",
            },
            "tactical_gameplan": {
                "home_team": {
                    "team": h_tactics["team_name"],
                    "best_offensive_action": h_opt["best_offensive_action"],
                    "best_defensive_action": h_opt["best_defensive_action"],
                    "adjusted_offensive_weights": h_opt["adjusted_offensive_weights"],
                    "adjusted_defensive_weights": h_opt["adjusted_defensive_weights"],
                },
                "away_team": {
                    "team": a_tactics["team_name"],
                    "best_offensive_action": a_opt["best_offensive_action"],
                    "best_defensive_action": a_opt["best_defensive_action"],
                    "adjusted_offensive_weights": a_opt["adjusted_offensive_weights"],
                    "adjusted_defensive_weights": a_opt["adjusted_defensive_weights"],
                },
            },
            "lineup_predicted_minutes": {
                "home_team": h_lineup_raw[:8],
                "away_team": a_lineup_raw[:8],
            },
        }
