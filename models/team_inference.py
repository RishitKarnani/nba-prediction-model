"""
models/team_inference.py
────────────────────────
End-to-End Team Game Outcome & Scoring Margin Prediction Engine.

Integrates:
  1. 15-Man Active Rosters with Model-Predicted Playing Time (LightGBM).
  2. Tactical Action Synergy Matrix (5 Offensive Actions vs 5 Defensive Coverages).
  3. Non-Zero Weight Rebalancing based on Opponent-Specific Win Expectation.
  4. Winner, Win Probability, Projected Scores, Point Spread Margin, and Tactical Gameplan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """

    def __init__(self, auto_load_simulation: bool = True):
        self.sim_results = None
        if auto_load_simulation and SIMULATION_RESULTS_PATH.exists():
            with open(SIMULATION_RESULTS_PATH, "r") as f:
                self.sim_results = json.load(f)
        else:
            self.sim_results = run_full_league_simulation(sims_per_matchup=10)

        if ROSTER_CACHE_PATH.exists():
            with open(ROSTER_CACHE_PATH, "r") as f:
                self.rosters = json.load(f)
        else:
            self.rosters = generate_all_team_predicted_minutes()

        self.baselines = load_team_baseline_ratings()

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        n_simulations: int = 10,
    ) -> Dict[str, Any]:
        """
        Predict game outcome, winning margin, final score, and tactical gameplans.
        """
        home_abbrev = resolve_team_abbrev(home_team)
        away_abbrev = resolve_team_abbrev(away_team)

        if home_abbrev == away_abbrev:
            raise ValueError("Home and Away teams cannot be the same.")

        fixture_key = f"{away_abbrev}_at_{home_abbrev}"

        # Check pre-computed cache
        cached = self.sim_results["matchups"].get(fixture_key)
        if cached and n_simulations == 10:
            matchup_data = cached
        else:
            # Dynamic on-the-fly calculation
            h_lineup = self.rosters[home_abbrev]["lineup"]
            a_lineup = self.rosters[away_abbrev]["lineup"]
            h_lineup_adj = calculate_lineup_impact(h_lineup)
            a_lineup_adj = calculate_lineup_impact(a_lineup)

            h_net = self.baselines[home_abbrev]["base_ortg"] - self.baselines[home_abbrev]["base_drtg"] + h_lineup_adj + 2.8
            a_net = self.baselines[away_abbrev]["base_ortg"] - self.baselines[away_abbrev]["base_drtg"] + a_lineup_adj
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

            home_wins = 0
            home_pts_list = []
            away_pts_list = []
            margins = []

            for _ in range(n_simulations):
                res = simulate_game(
                    home_team=home_abbrev,
                    away_team=away_abbrev,
                    home_baselines=self.baselines[home_abbrev],
                    away_baselines=self.baselines[away_abbrev],
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

            matchup_data = {
                "fixture": fixture_key,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_team_name": h_tactics["team_name"],
                "away_team_name": a_tactics["team_name"],
                "simulations": n_simulations,
                "home_win_probability": home_win_prob,
                "away_win_probability": round(1.0 - home_win_prob, 3),
                "predicted_winner": pred_winner,
                "predicted_margin": avg_margin,
                "predicted_home_pts": round(float(sum(home_pts_list) / len(home_pts_list)), 1),
                "predicted_away_pts": round(float(sum(away_pts_list) / len(away_pts_list)), 1),
                "home_tactical_gameplan": h_opt,
                "away_tactical_gameplan": a_opt,
            }

        # Build clean structured presentation response
        winner_name = (
            matchup_data["home_team_name"]
            if matchup_data["predicted_winner"] == matchup_data["home_team"]
            else matchup_data["away_team_name"]
        )
        margin_val = abs(matchup_data["predicted_margin"])
        margin_str = f"+{margin_val:.1f} ({matchup_data['predicted_winner']})"

        return {
            "fixture": matchup_data["fixture"],
            "home_team": matchup_data["home_team"],
            "away_team": matchup_data["away_team"],
            "home_team_name": matchup_data["home_team_name"],
            "away_team_name": matchup_data["away_team_name"],
            "prediction": {
                "predicted_winner": winner_name,
                "winner_abbreviation": matchup_data["predicted_winner"],
                "home_win_probability": matchup_data["home_win_probability"],
                "away_win_probability": matchup_data["away_win_probability"],
                "predicted_score": {
                    matchup_data["home_team"]: matchup_data["predicted_home_pts"],
                    matchup_data["away_team"]: matchup_data["predicted_away_pts"],
                },
                "predicted_margin": margin_str,
                "implied_spread": f"{matchup_data['predicted_winner']} -{margin_val:.1f}",
            },
            "tactical_gameplan": {
                "home_team": {
                    "team": matchup_data["home_team_name"],
                    "best_offensive_action": matchup_data["home_tactical_gameplan"]["best_offensive_action"],
                    "best_defensive_action": matchup_data["home_tactical_gameplan"]["best_defensive_action"],
                    "adjusted_offensive_weights": matchup_data["home_tactical_gameplan"]["adjusted_offensive_weights"],
                    "adjusted_defensive_weights": matchup_data["home_tactical_gameplan"]["adjusted_defensive_weights"],
                },
                "away_team": {
                    "team": matchup_data["away_team_name"],
                    "best_offensive_action": matchup_data["away_tactical_gameplan"]["best_offensive_action"],
                    "best_defensive_action": matchup_data["away_tactical_gameplan"]["best_defensive_action"],
                    "adjusted_offensive_weights": matchup_data["away_tactical_gameplan"]["adjusted_offensive_weights"],
                    "adjusted_defensive_weights": matchup_data["away_tactical_gameplan"]["adjusted_defensive_weights"],
                },
            },
            "lineup_predicted_minutes": {
                "home_team": self.rosters[home_abbrev]["lineup"][:8],
                "away_team": self.rosters[away_abbrev]["lineup"][:8],
            },
        }
