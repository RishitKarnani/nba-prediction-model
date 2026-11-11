"""
Command-line interface for querying the Version 1 NBA Player Performance & Probability Engine.

Strictly outputs player performance predictions, empirical outcome distributions,
and Over/Under/Push probabilities for user-supplied thresholds (no odds or EV).

Examples:
    python scripts/query_player.py --player "Stephen Curry" --opponent LAL --line PTS:25.5 --line 3PM:3.5
    python scripts/query_player.py --player "LeBron James" --opponent GSW --away --date 2026-11-01 --line PTS:20.5 PTS:25.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.inference import PlayerProbabilityEngine
from models.residuals import TARGETS, canonical_stat_name


def parse_line_specs(specs: List[str]) -> Dict[str, List[float]]:
    """Parse CLI line specs of the form STAT:LINE[,LINE2,...] into dict[stat, list[float]]."""
    lines_by_stat: Dict[str, List[float]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --line format '{spec}'. Expected STAT:VALUE (e.g. PTS:25.5).")
        raw_stat, raw_vals = spec.split(":", 1)
        stat = canonical_stat_name(raw_stat)
        vals = [float(v.strip()) for v in raw_vals.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"No threshold values provided in '{spec}'.")
        lines_by_stat.setdefault(stat, []).extend(vals)
    return lines_by_stat


def _dist_to_dict(raw_d: object) -> dict:
    """Extract clean summary dictionary from OutcomeDistribution or plain dict."""
    if isinstance(raw_d, dict):
        return raw_d
    lo, hi = raw_d.interval(0.90)
    return {
        "point_pred": round(float(raw_d.predicted_value), 2),
        "sim_mean": round(float(raw_d.distribution_mean), 2),
        "sim_median": round(float(raw_d.distribution_median), 2),
        "sim_std": round(float(raw_d.distribution_std), 2),
        "interval_90": [round(float(lo), 2), round(float(hi), 2)],
        "tier": str(raw_d.fallback_tier),
        "n_support": int(raw_d.n_support),
    }


def format_report(result: dict) -> str:
    """Render a clean human-readable performance and probability summary."""
    meta = result.get("metadata", {})
    venue = "vs." if meta.get("is_home", 1) == 1 else "@"
    role = "Starter" if result.get("is_starter_context") else "Bench"
    header = (
        f"\n{meta.get('player_name', 'Player')} ({meta.get('team_abbreviation', '')}) "
        f"{venue} {meta.get('opponent_abbrev', '')} | Date: {meta.get('game_date', '')} | "
        f"Predicted Minutes: {result['predicted_minutes']:.2f} ({role})"
    )
    lines = [header, "=" * len(header.strip())]
    lines.append(f"{'Stat':<5} {'Point Pred':>10} {'Sim Mean':>10} {'Std':>7} {'90% Range':>14} {'Tier (n_support)':>28}")
    lines.append("-" * 79)

    dists = result.get("distributions", {})
    for stat in TARGETS:
        d = _dist_to_dict(dists[stat]) if stat in dists else {}
        ci = d.get("interval_90", [0.0, 0.0])
        ci_str = f"[{ci[0]:.0f}, {ci[1]:.0f}]"
        tier_str = f"{d.get('tier', d.get('fallback_tier', 'unknown'))} ({d.get('n_support', 0)})"
        lines.append(
            f"{stat:<5} {result['predicted_stats'].get(stat, 0.0):>10.2f} "
            f"{d.get('sim_mean', 0.0):>10.2f} {d.get('sim_std', 0.0):>7.2f} "
            f"{ci_str:>14} {tier_str:>28}"
        )

    line_probs = result.get("line_probabilities", {})
    if line_probs:
        lines.append("\nThreshold Probabilities (10,000 simulations):")
        lines.append(f"{'Stat':<5} {'Line':>6} {'P(Over)':>9} {'P(Under)':>9} {'P(Push)':>9} {'Status'}")
        lines.append("-" * 68)
        for stat in TARGETS:
            for item in line_probs.get(stat, []):
                push_str = f"{item['prob_push']*100:7.2f}%" if item["is_integer_line"] else "    0.00%"
                lines.append(
                    f"{stat:<5} {item['line']:>6.1f} "
                    f"{item['prob_over']*100:>8.2f}% "
                    f"{item['prob_under']*100:>8.2f}% "
                    f"{push_str:>9} "
                    f"{item['calibration_status']}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query NBA Player Performance & Probability Engine (v1)."
    )
    parser.add_argument("--player", required=True, help="Player full/partial name or PLAYER_ID.")
    parser.add_argument("--opponent", required=True, help="Opponent team abbreviation, name, or TEAM_ID.")
    parser.add_argument("--away", action="store_true", help="Set road matchup (default is home).")
    parser.add_argument("--date", default=None, help="Target game date YYYY-MM-DD (optional).")
    parser.add_argument(
        "--line",
        action="extend",
        nargs="+",
        default=[],
        help="Threshold lines formatted as STAT:VALUE[,VALUE2] (e.g. PTS:25.5 AST:6.5).",
    )
    parser.add_argument("--json", action="store_true", help="Output full JSON response.")
    args = parser.parse_args()

    lines_by_stat = parse_line_specs(args.line) if args.line else None
    engine = PlayerProbabilityEngine()
    result = engine.predict_player(
        player=args.player,
        opponent=args.opponent,
        is_home=0 if args.away else 1,
        game_date=args.date,
        lines_by_stat=lines_by_stat,
    )
    if args.json:
        serializable = dict(result)
        serializable["distributions"] = {
            k: _dist_to_dict(v) for k, v in result.get("distributions", {}).items()
        }
        print(json.dumps(serializable, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
