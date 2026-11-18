"""
src/weight_optimizer.py
───────────────────────
Implements the non-zero bounded weight adjustment algorithm for team tactical actions.
Adjusts the 5 offensive actions and 5 defensive actions in proportion to their
expected win probability against each opponent, strictly enforcing w_k > 0 for all actions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

# Minimum baseline probability floor for any action (4% floor guarantees strictly w > 0)
MIN_ACTION_WEIGHT_FLOOR = 0.04


def adjust_action_weights_ratio(
    win_expectations: List[float],
    min_floor: float = MIN_ACTION_WEIGHT_FLOOR,
    temperature: float = 1.5,
) -> List[float]:
    """
    Adjust weights in proportion to expected win probability without allowing any weight to be 0.

    Parameters
    ----------
    win_expectations : list of 5 floats
        Estimated win probability when employing each of the 5 actions.
    min_floor : float, default 0.04
        Strict lower bound for each action's weight (0.04 = 4% minimum).
    temperature : float, default 1.5
        Sensitivity parameter for scaling win differences into weight shifts.

    Returns
    -------
    list of 5 floats
        Adjusted weights summing strictly to 1.0, with each weight >= min_floor.
    """
    n = len(win_expectations)
    if n != 5:
        raise ValueError(f"Expected 5 actions, got {n}")

    # Clip win expectations to prevent zero or negative values
    wins = np.clip(np.array(win_expectations, dtype=float), 0.01, 0.99)

    # Exponentiate with temperature to amplify meaningful tactical differences
    powered_wins = np.power(wins, temperature)
    normalized_ratios = powered_wins / np.sum(powered_wins)

    # Distribute the remaining probability mass above the floor
    remaining_mass = 1.0 - (n * min_floor)
    adjusted_weights = min_floor + (remaining_mass * normalized_ratios)

    # Ensure exact rounding and strict 1.0 sum
    rounded = [round(float(w), 4) for w in adjusted_weights]
    diff = 1.0 - sum(rounded)
    # Apply minuscule floating point difference to the highest weight action
    max_idx = int(np.argmax(rounded))
    rounded[max_idx] = round(rounded[max_idx] + diff, 4)

    return rounded


def optimize_team_tactical_weights(
    off_win_rates: List[float],
    def_win_rates: List[float],
    offensive_action_names: List[str],
    defensive_action_names: List[str],
) -> Dict[str, Any]:
    """
    Compute adjusted weights and identify best counter actions for a team.
    """
    adj_off_weights = adjust_action_weights_ratio(off_win_rates)
    adj_def_weights = adjust_action_weights_ratio(def_win_rates)

    best_off_idx = int(np.argmax(off_win_rates))
    best_def_idx = int(np.argmax(def_win_rates))

    return {
        "adjusted_offensive_weights": dict(zip(offensive_action_names, adj_off_weights)),
        "adjusted_defensive_weights": dict(zip(defensive_action_names, adj_def_weights)),
        "best_offensive_action": offensive_action_names[best_off_idx],
        "best_offensive_win_rate": round(float(off_win_rates[best_off_idx]), 4),
        "best_defensive_action": defensive_action_names[best_def_idx],
        "best_defensive_win_rate": round(float(def_win_rates[best_def_idx]), 4),
        "raw_off_weights_list": adj_off_weights,
        "raw_def_weights_list": adj_def_weights,
    }
