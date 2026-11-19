"""
src/injury_engine.py
────────────────────
Injury Impact Engine.

When a player is listed as OUT:
  1. Identify player in the 15-man lineup by fuzzy name match.
  2. Set their allocated_minutes to 0.
  3. Redistribute their vacated minutes proportionally to remaining active players.
  4. Re-normalise total to exactly 240.0 minutes.
  5. Subtract the injured player's estimated per-game scoring contribution.
  6. Redistribute that lost scoring proportionally (by minute share) to active players.
  7. Recompute the team's effective offensive-rating adjustment.

Injury model (exact user spec):
  "Take the 14-man lineup offensive average minus Luka's points,
   and those points are redistributed to all remaining players."

Usage
-----
  from src.injury_engine import apply_injuries

  updated_lineup, ortg_adj, summary = apply_injuries(
      lineup=lal_lineup,
      injured_names=["Luka Doncic"],
  )
"""

from __future__ import annotations

import copy
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pace constant: NBA average possessions per game ≈ 100 (used to convert PPG → ortg pts)
AVG_PACE = 100.0
TOTAL_MINUTES = 240.0  # 5 players × 48 minutes


def _fuzzy_match(name: str, candidates: List[str], threshold: float = 0.72) -> Optional[str]:
    """Return best fuzzy match from candidates, or None if below threshold."""
    name_lower = name.lower().strip()
    best_score = 0.0
    best_match = None
    for c in candidates:
        score = SequenceMatcher(None, name_lower, c.lower().strip()).ratio()
        # Also check last-name exact match
        last_name = name_lower.split()[-1]
        c_last = c.lower().strip().split()[-1]
        if last_name == c_last:
            score = max(score, 0.90)
        if score > best_score:
            best_score = score
            best_match = c
    if best_score >= threshold:
        return best_match
    return None


def apply_injuries(
    lineup: List[Dict],
    injured_names: List[str],
    game_pace: float = AVG_PACE,
) -> Tuple[List[Dict], float, Dict]:
    """
    Apply an injury list to a team's predicted-minutes lineup.

    Parameters
    ----------
    lineup : list of dicts
        Each dict must have at minimum:
          - 'player_name'       : str
          - 'allocated_minutes' : float  (sums to 240.0)
          - 'historical_pts_per_game' : float  (player's season PPG)
        Typically the output of roster_minutes.py's lineup list.
    injured_names : list of str
        Names of players who are OUT (e.g. ["Luka Doncic", "Anthony Davis"]).
    game_pace : float
        Estimated possessions per game (default 100).

    Returns
    -------
    updated_lineup : list of dicts
        Lineup with minutes/points redistributed; injured players have minutes=0.
    ortg_adjustment : float
        Estimated change in offensive rating (always ≤ 0 when players are out).
    summary : dict
        Human-readable breakdown of what was removed and redistributed.
    """
    if not injured_names:
        return lineup, 0.0, {"injuries": [], "pts_lost": 0.0, "ortg_adjustment": 0.0}

    lineup = copy.deepcopy(lineup)
    player_names = [p["player_name"] for p in lineup]

    injured_indices: List[int] = []
    unmatched: List[str] = []

    for inj_name in injured_names:
        matched = _fuzzy_match(inj_name, player_names)
        if matched is None:
            logger.warning(f"[InjuryEngine] '{inj_name}' not found in lineup — skipped.")
            unmatched.append(inj_name)
            continue
        idx = player_names.index(matched)
        if idx in injured_indices:
            continue
        injured_indices.append(idx)
        logger.info(f"[InjuryEngine] Matched '{inj_name}' → '{matched}' (idx={idx})")

    if not injured_indices:
        return lineup, 0.0, {
            "injuries": [],
            "pts_lost": 0.0,
            "ortg_adjustment": 0.0,
            "unmatched": unmatched,
            "note": "No matched players found in roster.",
        }

    # ── Step 1: Extract injured player stats ──────────────────────────────────
    injury_records = []
    total_vacated_minutes = 0.0
    total_pts_lost = 0.0

    for idx in injured_indices:
        p = lineup[idx]
        player_min = p.get("allocated_minutes", 0.0)
        player_ppg = p.get("historical_pts_per_game", 0.0)

        # Estimated per-game scoring contribution (scaled by minute share)
        # A player who plays avg 36 min in a 240-min game contributes (36/48)*PPG per game
        # But since allocated_minutes are within the team's 240-min budget:
        min_share = player_min / TOTAL_MINUTES
        pts_contribution = player_ppg * (player_min / 48.0)   # prorated to 48-min game

        injury_records.append({
            "player_name":    p["player_name"],
            "vacated_minutes": player_min,
            "ppg":            player_ppg,
            "pts_contribution": round(pts_contribution, 2),
        })
        total_vacated_minutes += player_min
        total_pts_lost += pts_contribution

        # Mark player as OUT
        lineup[idx]["allocated_minutes"] = 0.0
        lineup[idx]["minute_ratio"]      = 0.0
        lineup[idx]["status"]            = "OUT"

    # ── Step 2: Find active players and redistribute minutes ──────────────────
    active_indices = [i for i in range(len(lineup)) if i not in injured_indices]

    if not active_indices:
        logger.error("[InjuryEngine] No active players left — cannot redistribute.")
        return lineup, -50.0, {"error": "All players injured"}

    active_minutes_sum = sum(lineup[i]["allocated_minutes"] for i in active_indices)

    if active_minutes_sum <= 0:
        # Fallback: assign equal minutes
        equal_min = TOTAL_MINUTES / len(active_indices)
        for i in active_indices:
            lineup[i]["allocated_minutes"] = equal_min
        active_minutes_sum = TOTAL_MINUTES

    # Redistribute vacated minutes proportionally to active players
    for i in active_indices:
        orig_min = lineup[i]["allocated_minutes"]
        extra_min = (orig_min / active_minutes_sum) * total_vacated_minutes
        lineup[i]["allocated_minutes"] = orig_min + extra_min

    # Re-normalise to exactly 240.0
    new_total = sum(lineup[i]["allocated_minutes"] for i in active_indices)
    scale = TOTAL_MINUTES / new_total if new_total > 0 else 1.0
    for i in active_indices:
        lineup[i]["allocated_minutes"] = round(lineup[i]["allocated_minutes"] * scale, 2)
        lineup[i]["minute_ratio"]      = round(lineup[i]["allocated_minutes"] / TOTAL_MINUTES, 4)

    # ── Step 3: Redistribute lost points scoring to active players ────────────
    # Active players absorb a fraction of the star's points (usage redistribution)
    # Weighted by their new minute share
    new_active_min_sum = sum(lineup[i]["allocated_minutes"] for i in active_indices)
    pts_per_active: Dict[int, float] = {}
    for i in active_indices:
        share = lineup[i]["allocated_minutes"] / new_active_min_sum
        pts_per_active[i] = round(total_pts_lost * share, 2)
        # Update effective PPG (approximate)
        old_ppg = lineup[i].get("historical_pts_per_game", 0.0)
        # Distribute extra points as a PPG bump (prorated to 48 min)
        pts_bump_ppg = pts_per_active[i] / (lineup[i]["allocated_minutes"] / 48.0) if lineup[i]["allocated_minutes"] > 0 else 0
        lineup[i]["effective_ppg"] = round(old_ppg + pts_bump_ppg * 0.60, 2)  # 60% of star pts realistically absorbed

    # ── Step 4: Compute offensive rating adjustment ───────────────────────────
    # ortg adjustment ≈ -(unreplaced pts lost / game_pace) * 100
    # Assumption: 60% of the star's pts get absorbed by active players, 40% are truly lost
    unreplaced_pts = total_pts_lost * 0.40
    ortg_adjustment = -(unreplaced_pts / game_pace) * 100.0

    # ── Step 5: Build summary ─────────────────────────────────────────────────
    summary = {
        "injuries": injury_records,
        "unmatched_names": unmatched,
        "total_vacated_minutes": round(total_vacated_minutes, 1),
        "total_pts_lost": round(total_pts_lost, 2),
        "unreplaced_pts": round(unreplaced_pts, 2),
        "ortg_adjustment": round(ortg_adjustment, 2),
        "pts_redistributed_per_active": pts_per_active,
        "active_player_count": len(active_indices),
        "minutes_sanity_check": round(sum(p["allocated_minutes"] for p in lineup), 1),
    }

    logger.info(
        f"[InjuryEngine] {len(injured_indices)} player(s) OUT | "
        f"vacated_min={total_vacated_minutes:.1f} | "
        f"pts_lost={total_pts_lost:.1f} | "
        f"ortg_adj={ortg_adjustment:+.1f}"
    )

    return lineup, round(ortg_adjustment, 2), summary


def format_injury_report(team_abbrev: str, summary: Dict) -> str:
    """Format a human-readable injury impact report section."""
    lines = [f"\n[INJURY ADJUSTMENTS: {team_abbrev}]"]

    if not summary.get("injuries"):
        lines.append("  No injuries applied.")
        return "\n".join(lines)

    for rec in summary["injuries"]:
        lines.append(f"  ⛔  {rec['player_name']} — OUT")
        lines.append(f"       Minutes vacated     : {rec['vacated_minutes']:.1f} min → redistributed to {summary['active_player_count']} active players")
        lines.append(f"       Estimated pts/game  : {rec['pts_contribution']:.1f} pts")

    lines.append(f"  ─────────────────────────────────────────────────")
    lines.append(f"  Total pts lost (est.)     : {summary['total_pts_lost']:.1f} pts/game")
    lines.append(f"  Pts truly unreplaced (40%): {summary['unreplaced_pts']:.1f} pts/game")
    lines.append(f"  Offensive Rating Δ        : {summary['ortg_adjustment']:+.1f} pts (applied to simulation)")
    lines.append(f"  Minutes sanity check      : {summary['minutes_sanity_check']:.1f} / 240.0 ✓")

    if summary.get("unmatched_names"):
        lines.append(f"  ⚠️  Not found in roster    : {', '.join(summary['unmatched_names'])}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.roster_minutes import generate_all_team_predicted_minutes

    rosters = generate_all_team_predicted_minutes()
    lal_lineup = rosters["LAL"]["lineup"]

    print("=== LAL Lineup (original) ===")
    for p in lal_lineup[:6]:
        print(f"  {p['player_name']:<22} {p['allocated_minutes']:.1f} min  PPG:{p.get('historical_pts_per_game', 'N/A')}")

    updated, ortg_adj, summary = apply_injuries(lal_lineup, ["Luka Doncic"])

    print("\n=== LAL Lineup (Luka OUT) ===")
    for p in updated[:7]:
        status = "❌ OUT" if p.get("status") == "OUT" else ""
        print(f"  {p['player_name']:<22} {p['allocated_minutes']:.1f} min  {status}")

    print(format_injury_report("LAL", summary))
