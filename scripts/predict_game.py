"""
scripts/predict_game.py
───────────────────────
CLI for predicting NBA game outcomes, scoring margin, and optimal tactical gameplans.

Usage:
    python scripts/predict_game.py --home BOS --away DAL
    python scripts/predict_game.py --home LAL --away GSW --injuries-home "Luka Doncic" --injuries-away "Stephen Curry"
    python scripts/predict_game.py --home MIA --away OKC --apply-form --json
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
from src.injury_engine import format_injury_report


def format_form_section(abbrev: str, form: dict, label: str) -> str:
    if not form:
        return ""
    lines = [f"\n[FORM ANALYSIS: {abbrev} ({label})]"]
    tier = form.get("tier", "N/A")
    lines.append(f"  Current Form          : {tier}")
    lines.append(f"  Last 5  Net Rating    : {form.get('roll5_net', 0):+.1f}")
    lines.append(f"  Last 10 Net Rating    : {form.get('roll10_net', 0):+.1f}")
    lines.append(f"  Season  Net Rating    : {form.get('season_net', 0):+.1f}")
    lines.append(f"  Form-Weighted Net     : {form.get('form_net', 0):+.1f}  ← used in simulation")
    lines.append(f"  Form Win%             : {form.get('form_win_pct', 0)*100:.1f}%")
    ortg_adj = form.get("ortg_adjustment", 0)
    drtg_adj = form.get("drtg_adjustment", 0)
    lines.append(f"  Ortg adjustment (Δ)   : {ortg_adj:+.1f} pts vs raw season")
    lines.append(f"  Drtg adjustment (Δ)   : {drtg_adj:+.1f} pts vs raw season")
    return "\n".join(lines)


def format_report(res: dict) -> str:
    pred = res["prediction"]
    t_home = res["tactical_gameplan"]["home_team"]
    t_away = res["tactical_gameplan"]["away_team"]
    h = res["home_team"]
    a = res["away_team"]

    lines = []
    lines.append("=" * 76)
    lines.append(f"  NBA GAME PREDICTION & TACTICAL ANALYSIS: {a} @ {h}")
    lines.append(f"  {res['away_team_name']} at {res['home_team_name']}")
    if res.get("form_applied"):
        lines.append(f"  [Form weighting ACTIVE: 50% Last-5 | 30% Last-10 | 20% Season]")
    lines.append("=" * 76)

    # 1. Form analysis
    h_form = res.get("form", {}).get(h, {})
    a_form = res.get("form", {}).get(a, {})
    if h_form:
        lines.append(format_form_section(h, h_form, "HOME"))
    if a_form:
        lines.append(format_form_section(a, a_form, "AWAY"))

    # 2. Injury adjustments
    h_inj = res.get("injuries", {}).get(h, {})
    a_inj = res.get("injuries", {}).get(a, {})
    if h_inj and h_inj.get("injuries"):
        lines.append(format_injury_report(h, h_inj))
    if a_inj and a_inj.get("injuries"):
        lines.append(format_injury_report(a, a_inj))

    # 3. Prediction Outcome
    lines.append("\n[GAME OUTCOME PREDICTION]")
    lines.append(f"  Predicted Winner    : {pred['predicted_winner']} ({pred['winner_abbreviation']})")
    lines.append(f"  Win Probability     : {h} {pred['home_win_probability']*100:.1f}% | {a} {pred['away_win_probability']*100:.1f}%")
    score_home = pred["predicted_score"][h]
    score_away = pred["predicted_score"][a]
    lines.append(f"  Predicted Score     : {h} {score_home:.1f}  -  {a} {score_away:.1f}")
    lines.append(f"  Predicted Margin    : {pred['predicted_margin']}")
    lines.append(f"  Implied Spread Line : {pred['implied_spread']}")

    # 4. Tactical Gameplan - Home Team
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

    # 5. Tactical Gameplan - Away Team
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

    # 6. Lineup minutes
    lines.append(f"\n[MODEL-PREDICTED LINEUP MINUTES (LIGHTGBM)]")
    lines.append(f"  {h} Top Rotation:")
    for p in res["lineup_predicted_minutes"]["home_team"][:5]:
        status = " ❌OUT" if p.get("status") == "OUT" else ""
        lines.append(f"    * {p['player_name']:<22} [{p.get('role','?'):<9}] : {p['allocated_minutes']:>4.1f} min ({p.get('minute_ratio',0)*100:.1f}%){status}")
    lines.append(f"  {a} Top Rotation:")
    for p in res["lineup_predicted_minutes"]["away_team"][:5]:
        status = " ❌OUT" if p.get("status") == "OUT" else ""
        lines.append(f"    * {p['player_name']:<22} [{p.get('role','?'):<9}] : {p['allocated_minutes']:>4.1f} min ({p.get('minute_ratio',0)*100:.1f}%){status}")

    lines.append("\n" + "=" * 76)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NBA Team Outcome, Scoring Margin & Tactical Action Prediction")
    parser.add_argument("--home", required=True,  help="Home team abbreviation or name (e.g. BOS, 'Boston Celtics')")
    parser.add_argument("--away", required=True,  help="Away team abbreviation or name (e.g. DAL, 'Dallas Mavericks')")
    parser.add_argument("--injuries-home", default="", help="Comma-separated list of injured HOME players (e.g. 'Luka Doncic,Anthony Davis')")
    parser.add_argument("--injuries-away", default="", help="Comma-separated list of injured AWAY players")
    parser.add_argument("--no-form",   action="store_true", help="Disable form-factor weighting (use raw season baselines)")
    parser.add_argument("--json",      action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    inj_home = [n.strip() for n in args.injuries_home.split(",") if n.strip()]
    inj_away = [n.strip() for n in args.injuries_away.split(",") if n.strip()]
    apply_form = not args.no_form

    engine = TeamPredictionEngine(apply_form=apply_form)
    result = engine.predict_game(
        home_team=args.home,
        away_team=args.away,
        injuries_home=inj_home,
        injuries_away=inj_away,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
