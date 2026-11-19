"""
src/matchup_simulator.py
────────────────────────
Monte Carlo Matchup Simulator for all 30 NBA Teams.

Simulates each NBA team against every other NBA team 10 times at Home and 10 times Away
(30 teams x 29 opponents = 870 fixtures x 10 iterations = 8,700 simulated games).

Incorporates:
  1. 15-man active lineups with minutes exclusively predicted by the LightGBM minutes model.
  2. Tactical action synergy matrix (5 Offensive Actions vs 5 Defensive Coverages).
  3. Bounded non-zero weight adjustments (w_k >= 0.04 > 0) in the ratio of expected win.
  4. Game margins, predicted scores, and optimal action identification.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.roster_minutes import generate_all_team_predicted_minutes
from src.tactical_registry import (
    TEAM_TACTICAL_ACTIONS,
    get_action_delta,
)
from src.weight_optimizer import adjust_action_weights_ratio, optimize_team_tactical_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SIMULATION_RESULTS_PATH = ROOT_DIR / "artefacts" / "matchup_simulation_results.json"
TEAM_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "team_features.parquet"


def load_team_baseline_ratings() -> Dict[str, Dict[str, float]]:
    """
    Extract latest baseline rolling metrics for each team:
    Offensive Rating, Defensive Rating, Pace.
    """
    if not TEAM_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Team features not found at {TEAM_FEATURES_PATH}")

    df = pd.read_parquet(TEAM_FEATURES_PATH)
    df = df.sort_values("GAME_DATE")
    latest = df.groupby("TEAM_ABBREVIATION").last()

    baselines: Dict[str, Dict[str, float]] = {}
    for abbrev in TEAM_TACTICAL_ACTIONS.keys():
        if abbrev in latest.index:
            row = latest.loc[abbrev]
            ortg = float(row.get("ROLL20_OFF_RATING", 112.0))
            drtg = float(row.get("ROLL20_DEF_RATING", 112.0))
            raw_pace = float(row.get("ROLL20_PACE", 20.0))
            pace = raw_pace * 5.0 if raw_pace < 50.0 else raw_pace
            if pace < 85.0 or pace > 115.0:
                pace = 100.0
        else:
            ortg, drtg, pace = 112.0, 112.0, 100.0

        baselines[abbrev] = {
            "base_ortg": round(ortg, 2),
            "base_drtg": round(drtg, 2),
            "base_pace": round(pace, 2),
        }
    return baselines


def calculate_lineup_impact(lineup: List[Dict[str, Any]]) -> float:
    """
    Calculate the net rating adjustment from the 15-man lineup,
    weighted strictly by their model-predicted minute allocation.
    """
    net_adj = 0.0
    for p in lineup:
        ratio = p.get("minute_ratio", 1.0 / 15.0)
        pm = p.get("historical_plus_minus", 0.0)
        net_adj += ratio * pm * 0.35
    return float(np.clip(net_adj, -4.5, 4.5))


def calculate_tactical_action_expectations(
    team_a_abbrev: str,
    team_b_abbrev: str,
    baseline_margin: float,
    is_home_a: bool = True,
    scale_factor: float = 8.5,
) -> Tuple[List[float], List[float]]:
    """
    Calculate the expected win probability for each of Team A's 5 offensive
    and 5 defensive actions against Team B's schemes.
    """
    t_a = TEAM_TACTICAL_ACTIONS[team_a_abbrev]
    t_b = TEAM_TACTICAL_ACTIONS[team_b_abbrev]

    a_off_arch = t_a["action_archetypes_off"]
    a_def_arch = t_a["action_archetypes_def"]
    b_off_arch = t_b["action_archetypes_off"]
    b_def_arch = t_b["action_archetypes_def"]

    # 1. Offensive Actions Marginal Impact
    # How each action i performs against B's default defensive coverage (uniform 0.20)
    off_deltas = []
    for i in range(5):
        delta_i = sum(0.20 * get_action_delta(a_off_arch[i], b_def_arch[j]) for j in range(5))
        off_deltas.append(delta_i)

    avg_off_delta = float(np.mean(off_deltas))
    # Margins under each action: higher is better for Team A
    off_margins = [baseline_margin + (d - avg_off_delta) for d in off_deltas]
    off_win_expectations = [
        float(1.0 / (1.0 + np.exp(-m / scale_factor))) for m in off_margins
    ]

    # 2. Defensive Actions Marginal Impact
    # How each defensive action j suppresses B's offensive actions (uniform 0.20)
    # Opponent delta: negative delta for B is positive for A!
    def_deltas = []
    for j in range(5):
        opp_delta_j = sum(0.20 * get_action_delta(b_off_arch[k], a_def_arch[j]) for k in range(5))
        def_deltas.append(-opp_delta_j)  # Invert so higher means better defense for A

    avg_def_delta = float(np.mean(def_deltas))
    def_margins = [baseline_margin + (d - avg_def_delta) for d in def_deltas]
    def_win_expectations = [
        float(1.0 / (1.0 + np.exp(-m / scale_factor))) for m in def_margins
    ]

    return off_win_expectations, def_win_expectations


def simulate_game(
    home_team: str,
    away_team: str,
    home_baselines: Dict[str, float],
    away_baselines: Dict[str, float],
    home_lineup_adj: float,
    away_lineup_adj: float,
    home_off_weights: List[float],
    home_def_weights: List[float],
    away_off_weights: List[float],
    away_def_weights: List[float],
    hca_points: float = 2.8,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Simulate a single NBA game between Home and Away teams.
    """
    rng = np.random.default_rng(seed)

    # 1. Expected Pace
    avg_pace = (home_baselines["base_pace"] * away_baselines["base_pace"]) / 100.0
    game_pace = float(rng.normal(avg_pace, 2.2))
    game_pace = max(88.0, min(112.0, game_pace))

    # 2. Tactical Delta
    home_tactics = TEAM_TACTICAL_ACTIONS[home_team]
    away_tactics = TEAM_TACTICAL_ACTIONS[away_team]

    h_off_arch = home_tactics["action_archetypes_off"]
    h_def_arch = home_tactics["action_archetypes_def"]
    a_off_arch = away_tactics["action_archetypes_off"]
    a_def_arch = away_tactics["action_archetypes_def"]

    delta_h_off = sum(
        home_off_weights[i] * away_def_weights[j] * get_action_delta(h_off_arch[i], a_def_arch[j])
        for i in range(5) for j in range(5)
    )
    delta_a_off = sum(
        away_off_weights[i] * home_def_weights[j] * get_action_delta(a_off_arch[i], h_def_arch[j])
        for i in range(5) for j in range(5)
    )

    # 3. Opponent Defensive Impact & Net Offensive Ratings
    league_drtg = 112.0
    def_impact_home = (away_baselines.get("base_drtg", league_drtg) - league_drtg) * 0.75
    def_impact_away = (home_baselines.get("base_drtg", league_drtg) - league_drtg) * 0.75

    ortg_home = (
        home_baselines["base_ortg"]
        + def_impact_home
        + home_lineup_adj
        + delta_h_off
        + (hca_points / 2.0)
        + rng.normal(0, 4.0)
    )
    ortg_away = (
        away_baselines["base_ortg"]
        + def_impact_away
        + away_lineup_adj
        + delta_a_off
        - (hca_points / 2.0)
        + rng.normal(0, 4.0)
    )

    # 4. Points Scored
    pts_home = int(np.round(game_pace * (ortg_home / 100.0)))
    pts_away = int(np.round(game_pace * (ortg_away / 100.0)))

    # Overtime resolution if tied
    if pts_home == pts_away:
        ot_winner = rng.choice(["home", "away"])
        if ot_winner == "home":
            pts_home += int(rng.integers(2, 8))
        else:
            pts_away += int(rng.integers(2, 8))

    margin = pts_home - pts_away
    winner = home_team if margin > 0 else away_team

    return {
        "home_team": home_team,
        "away_team": away_team,
        "pts_home": pts_home,
        "pts_away": pts_away,
        "margin": margin,
        "winner": winner,
        "game_pace": round(game_pace, 1),
    }


def run_full_league_simulation(
    sims_per_matchup: int = 10,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Run 10 Home and 10 Away game simulations for all 30 NBA teams against each other.
    Total = 870 matchups x 10 iterations = 8,700 games.
    """
    if not force_refresh and SIMULATION_RESULTS_PATH.exists():
        with open(SIMULATION_RESULTS_PATH, "r") as f:
            return json.load(f)

    logger.info("Initializing 30-team rosters and predicted minutes...")
    rotations = generate_all_team_predicted_minutes(force_refresh=False)
    baselines = load_team_baseline_ratings()

    lineup_impacts = {
        abbrev: calculate_lineup_impact(rotations[abbrev]["lineup"])
        for abbrev in TEAM_TACTICAL_ACTIONS.keys()
    }

    teams = list(TEAM_TACTICAL_ACTIONS.keys())
    total_matchup_pairs = len(teams) * (len(teams) - 1)  # 30 * 29 = 870
    logger.info(f"Simulating {total_matchup_pairs} directed fixtures ({sims_per_matchup} Home & Away = {total_matchup_pairs * sims_per_matchup} total games)...")

    matchup_records: Dict[str, Any] = {}
    total_games_simulated = 0

    for h_idx, home_tm in enumerate(teams):
        for away_tm in teams:
            if home_tm == away_tm:
                continue

            fixture_key = f"{away_tm}_at_{home_tm}"

            # Step 1: Baseline expected margin
            h_net = baselines[home_tm]["base_ortg"] - baselines[home_tm]["base_drtg"] + lineup_impacts[home_tm] + 2.8
            a_net = baselines[away_tm]["base_ortg"] - baselines[away_tm]["base_drtg"] + lineup_impacts[away_tm]
            expected_base_margin = h_net - a_net

            # Step 2: Calculate tactical action win expectations
            h_off_wins, h_def_wins = calculate_tactical_action_expectations(
                home_tm, away_tm, baseline_margin=expected_base_margin, is_home_a=True
            )
            a_off_wins, a_def_wins = calculate_tactical_action_expectations(
                away_tm, home_tm, baseline_margin=-expected_base_margin, is_home_a=False
            )

            # Step 3: Compute bounded non-zero weight adjustments (w_k >= 0.04)
            h_tactics = TEAM_TACTICAL_ACTIONS[home_tm]
            a_tactics = TEAM_TACTICAL_ACTIONS[away_tm]

            h_opt = optimize_team_tactical_weights(
                h_off_wins,
                h_def_wins,
                h_tactics["offensive_actions"],
                h_tactics["defensive_actions"],
            )
            a_opt = optimize_team_tactical_weights(
                a_off_wins,
                a_def_wins,
                a_tactics["offensive_actions"],
                a_tactics["defensive_actions"],
            )

            # Step 4: Run 10 simulations using the dynamically adjusted weights
            home_wins = 0
            home_pts_list = []
            away_pts_list = []
            margins = []

            for sim_i in range(sims_per_matchup):
                total_games_simulated += 1
                res = simulate_game(
                    home_team=home_tm,
                    away_team=away_tm,
                    home_baselines=baselines[home_tm],
                    away_baselines=baselines[away_tm],
                    home_lineup_adj=lineup_impacts[home_tm],
                    away_lineup_adj=lineup_impacts[away_tm],
                    home_off_weights=h_opt["raw_off_weights_list"],
                    home_def_weights=h_opt["raw_def_weights_list"],
                    away_off_weights=a_opt["raw_off_weights_list"],
                    away_def_weights=a_opt["raw_def_weights_list"],
                )

                home_pts_list.append(res["pts_home"])
                away_pts_list.append(res["pts_away"])
                margins.append(res["margin"])
                if res["winner"] == home_tm:
                    home_wins += 1

            home_win_prob = round(home_wins / sims_per_matchup, 3)
            avg_margin = round(float(np.mean(margins)), 1)
            avg_home_pts = round(float(np.mean(home_pts_list)), 1)
            avg_away_pts = round(float(np.mean(away_pts_list)), 1)
            pred_winner = home_tm if avg_margin >= 0 else away_tm

            matchup_records[fixture_key] = {
                "fixture": fixture_key,
                "home_team": home_tm,
                "away_team": away_tm,
                "home_team_name": h_tactics["team_name"],
                "away_team_name": a_tactics["team_name"],
                "simulations": sims_per_matchup,
                "home_win_probability": home_win_prob,
                "away_win_probability": round(1.0 - home_win_prob, 3),
                "predicted_winner": pred_winner,
                "predicted_margin": avg_margin,
                "predicted_home_pts": avg_home_pts,
                "predicted_away_pts": avg_away_pts,
                "home_tactical_gameplan": h_opt,
                "away_tactical_gameplan": a_opt,
            }

    logger.info(f"Successfully simulated {total_games_simulated} games across all {len(matchup_records)} matchups!")

    output_payload = {
        "metadata": {
            "total_teams": len(teams),
            "total_matchups": len(matchup_records),
            "simulations_per_matchup": sims_per_matchup,
            "total_games_simulated": total_games_simulated,
            "min_weight_floor": 0.04,
        },
        "matchups": matchup_records,
    }

    SIMULATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SIMULATION_RESULTS_PATH, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Results saved to {SIMULATION_RESULTS_PATH}")
    return output_payload


if __name__ == "__main__":
    results = run_full_league_simulation(sims_per_matchup=10, force_refresh=True)
    print("\nSimulation Completed Successfully!")
    print(f"Total fixtures: {results['metadata']['total_matchups']}")
    print(f"Total games: {results['metadata']['total_games_simulated']}")

    # Preview sample fixture: DAL @ BOS
    sample = results["matchups"].get("DAL_at_BOS")
    if sample:
        print("\n=== Sample Matchup: Dallas Mavericks @ Boston Celtics ===")
        print(f"Predicted Winner: {sample['predicted_winner']} (Win Prob: {sample['home_win_probability']*100:.1f}%)")
        print(f"Predicted Score: BOS {sample['predicted_home_pts']} - DAL {sample['predicted_away_pts']}")
        print(f"Predicted Margin: {sample['predicted_margin']} (BOS)")
        print(f"Boston Top Offensive Action: {sample['home_tactical_gameplan']['best_offensive_action']}")
        print(f"Boston Top Defensive Action: {sample['home_tactical_gameplan']['best_defensive_action']}")
        print("\nBoston Adjusted Offensive Weights (Non-Zero):")
        for act, w in sample['home_tactical_gameplan']['adjusted_offensive_weights'].items():
            print(f"  - {act}: {w*100:.1f}%")
        print("\nDallas Adjusted Offensive Weights (Non-Zero):")
        for act, w in sample['away_tactical_gameplan']['adjusted_offensive_weights'].items():
            print(f"  - {act}: {w*100:.1f}%")
