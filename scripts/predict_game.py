"""
scripts/predict_game.py
───────────────────────
CLI for predicting NBA game outcomes, scoring margin, and optimal tactical gameplans.

Usage:
    python scripts/predict_game.py --home BOS --away DAL
    python scripts/predict_game.py --home "Golden State Warriors" --away "Los Angeles Lakers"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.team_inference import TeamPredictionEngine


def format_report(res: dict) -> str:
    pred = res["prediction"]
    t_home = res["tactical_gameplan"]["home_team"]
    t_away = res["tactical_gameplan"]["away_team"]

    lines = []
    lines.append("=" * 76)
    lines.append(f"  NBA GAME PREDICTION & TACTICAL ANALYSIS: {res['away_team']} @ {res['home_team']}")
    lines.append(f"  {res['away_team_name']} at {res['home_team_name']}")
    lines.append("=" * 76)

    # 1. Prediction Outcome
    lines.append("\n[GAME OUTCOME PREDICTION]")
    lines.append(f"  Predicted Winner    : {pred['predicted_winner']} ({pred['winner_abbreviation']})")
    lines.append(f"  Win Probability     : {res['home_team']} {pred['home_win_probability']*100:.1f}% | {res['away_team']} {pred['away_win_probability']*100:.1f}%")
    score_home = pred['predicted_score'][res['home_team']]
    score_away = pred['predicted_score'][res['away_team']]
    lines.append(f"  Predicted Score     : {res['home_team']} {score_home:.1f}  -  {res['away_team']} {score_away:.1f}")
    lines.append(f"  Predicted Margin    : {pred['predicted_margin']}")
    lines.append(f"  Implied Spread Line : {pred['implied_spread']}")

    # 2. Tactical Gameplan - Home Team
    lines.append(f"\n[TACTICAL GAMEPLAN: {res['home_team_name'].upper()} (HOME)]")
    lines.append(f"  Top Offensive Counter : {t_home['best_offensive_action']}")
    lines.append(f"  Top Defensive Counter : {t_home['best_defensive_action']}")
    lines.append("  Adjusted Offensive Weights (Non-Zero):")
    for act, w in t_home["adjusted_offensive_weights"].items():
        bar = "■" * int(w * 50)
        lines.append(f"    - {act:<38} : {w*100:>5.1f}% | {bar}")
    lines.append("  Adjusted Defensive Weights (Non-Zero):")
    for act, w in t_home["adjusted_defensive_weights"].items():
        bar = "■" * int(w * 50)
        lines.append(f"    - {act:<38} : {w*100:>5.1f}% | {bar}")

    # 3. Tactical Gameplan - Away Team
    lines.append(f"\n[TACTICAL GAMEPLAN: {res['away_team_name'].upper()} (AWAY)]")
    lines.append(f"  Top Offensive Counter : {t_away['best_offensive_action']}")
    lines.append(f"  Top Defensive Counter : {t_away['best_defensive_action']}")
    lines.append("  Adjusted Offensive Weights (Non-Zero):")
    for act, w in t_away["adjusted_offensive_weights"].items():
        bar = "■" * int(w * 50)
        lines.append(f"    - {act:<38} : {w*100:>5.1f}% | {bar}")
    lines.append("  Adjusted Defensive Weights (Non-Zero):")
    for act, w in t_away["adjusted_defensive_weights"].items():
        bar = "■" * int(w * 50)
        lines.append(f"    - {act:<38} : {w*100:>5.1f}% | {bar}")

    # 4. Model-Predicted 15-Man Rotation (Top 8 displayed)
    lines.append(f"\n[MODEL-PREDICTED LINEUP MINUTES (LIGHTGBM)]")
    lines.append(f"  {res['home_team']} Top Rotation:")
    for p in res["lineup_predicted_minutes"]["home_team"][:5]:
        lines.append(f"    * {p['player_name']:<22} [{p['role']:<9}] : {p['allocated_minutes']:>4.1f} min ({p['minute_ratio']*100:.1f}%)")
    lines.append(f"  {res['away_team']} Top Rotation:")
    for p in res["lineup_predicted_minutes"]["away_team"][:5]:
        lines.append(f"    * {p['player_name']:<22} [{p['role']:<9}] : {p['allocated_minutes']:>4.1f} min ({p['minute_ratio']*100:.1f}%)")

    lines.append("\n" + "=" * 76)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NBA Team Outcome, Scoring Margin & Tactical Action Prediction")
    parser.add_argument("--home", required=True, help="Home team abbreviation or name (e.g. BOS, 'Boston Celtics')")
    parser.add_argument("--away", required=True, help="Away team abbreviation or name (e.g. DAL, 'Dallas Mavericks')")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    engine = TeamPredictionEngine()
    result = engine.predict_game(args.home, args.away)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
