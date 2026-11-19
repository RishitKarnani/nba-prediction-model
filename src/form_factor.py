"""
src/form_factor.py
──────────────────
Team Form Factor Engine.

Computes a form-weighted offensive/defensive rating and win% for each team
using the last 5 games (50%), last 10 games (30%), and full-season average (20%).

Formula:
  form_ortg      = 0.50 * ROLL5_OFF_RATING  + 0.30 * ROLL10_OFF_RATING  + 0.20 * OFF_RATING
  form_drtg      = 0.50 * ROLL5_DEF_RATING  + 0.30 * ROLL10_DEF_RATING  + 0.20 * DEF_RATING
  form_win_pct   = 0.50 * ROLL5_WIN_PCT      + 0.30 * ROLL10_WIN_PCT     + 0.20 * season_win_pct
  form_net       = form_ortg - form_drtg

Returns a dict keyed by 3-letter team abbreviation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np  # noqa: F401 (used in get_form_adjusted_baselines)
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
TEAM_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "team_features.parquet"

# Form weighting constants
W5 = 0.50      # Last 5 games weight
W10 = 0.30     # Last 10 games weight
WSEASON = 0.20 # Full season weight

# Form tier thresholds (net rating)
HOT_THRESHOLD = 5.0     # Net Rating ≥ +5.0
COLD_THRESHOLD = -3.0   # Net Rating ≤ -3.0


def _form_tier(form_net: float) -> str:
    if form_net >= HOT_THRESHOLD:
        return "🔥 HOT"
    elif form_net <= COLD_THRESHOLD:
        return "❄️  COLD"
    else:
        return "〰️  NEUTRAL"


def compute_team_form(season: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compute form-weighted ratings for all 30 teams.

    Parameters
    ----------
    season : str, optional
        NBA season string like '2024-25'. If None, uses the most recent data.

    Returns
    -------
    dict
        {
          'BOS': {
            'form_ortg': float,       # Form-weighted offensive rating
            'form_drtg': float,       # Form-weighted defensive rating
            'form_net':  float,       # form_ortg - form_drtg
            'form_win_pct': float,    # Form-weighted win percentage (0.0–1.0)
            'roll5_net': float,       # Raw last-5 net rating
            'roll10_net': float,      # Raw last-10 net rating
            'season_net': float,      # Raw season net rating
            'tier': str,              # 'HOT' | 'NEUTRAL' | 'COLD'
            'ortg_adjustment': float, # Δ from pure season baseline (form_ortg - OFF_RATING)
            'drtg_adjustment': float, # Δ from pure season baseline (form_drtg - DEF_RATING)
          },
          ...
        }
    """
    if not TEAM_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Team features parquet not found: {TEAM_FEATURES_PATH}")

    df = pd.read_parquet(TEAM_FEATURES_PATH)

    # Optionally filter to a specific season
    if season and "SEASON_ID" in df.columns:
        df = df[df["SEASON_ID"] == season]

    # Get the latest row per team (most recent game's rolling window is what we want)
    team_col = "TEAM_ABBREVIATION"
    if team_col not in df.columns:
        raise KeyError(f"Column '{team_col}' not found in team_features.parquet")

    # Sort by date if available
    date_col = None
    for c in ["GAME_DATE", "game_date", "DATE"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df = df.sort_values(date_col)

    # Take the most recent row per team
    latest = df.groupby(team_col).last().reset_index()

    form_data: Dict[str, Dict[str, Any]] = {}

    for _, row in latest.iterrows():
        abbrev = row[team_col]

        # Pull raw values, default to season avg if rolling unavailable
        ortg_season = float(row.get("OFF_RATING", 110.0))
        drtg_season = float(row.get("DEF_RATING", 110.0))

        ortg_r5  = float(row.get("ROLL5_OFF_RATING",  ortg_season))
        drtg_r5  = float(row.get("ROLL5_DEF_RATING",  drtg_season))
        ortg_r10 = float(row.get("ROLL10_OFF_RATING", ortg_season))
        drtg_r10 = float(row.get("ROLL10_DEF_RATING", drtg_season))

        win_season = float(row.get("WIN", 0.5))  # could be raw 0/1 — handled below
        win_r5     = float(row.get("ROLL5_WIN_PCT",  row.get("ROLL5_WIN",  win_season)))
        win_r10    = float(row.get("ROLL10_WIN_PCT", row.get("ROLL10_WIN", win_season)))
        win_pct_season = float(row.get("ROLL20_WIN_PCT", win_season))

        # Guard against 0/1 single-game WIN column vs actual pct
        if win_pct_season > 1.0:
            win_pct_season /= 100.0
        if win_r5 > 1.0:
            win_r5 /= 100.0
        if win_r10 > 1.0:
            win_r10 /= 100.0

        # Form-weighted aggregation
        form_ortg    = W5 * ortg_r5  + W10 * ortg_r10  + WSEASON * ortg_season
        form_drtg    = W5 * drtg_r5  + W10 * drtg_r10  + WSEASON * drtg_season
        form_net     = form_ortg - form_drtg
        form_win_pct = W5 * win_r5   + W10 * win_r10   + WSEASON * win_pct_season

        roll5_net  = ortg_r5  - drtg_r5
        roll10_net = ortg_r10 - drtg_r10
        season_net = ortg_season - drtg_season

        form_data[abbrev] = {
            "form_ortg":       round(form_ortg, 2),
            "form_drtg":       round(form_drtg, 2),
            "form_net":        round(form_net, 2),
            "form_win_pct":    round(form_win_pct, 4),
            "roll5_net":       round(roll5_net, 2),
            "roll10_net":      round(roll10_net, 2),
            "season_net":      round(season_net, 2),
            "ortg_season":     round(ortg_season, 2),
            "drtg_season":     round(drtg_season, 2),
            "tier":            _form_tier(form_net),
            "ortg_adjustment": round(form_ortg - ortg_season, 2),  # Δ vs raw season baseline
            "drtg_adjustment": round(form_drtg - drtg_season, 2),
        }

    logger.info(f"Form factor computed for {len(form_data)} teams (season={season or 'latest'})")
    return form_data


def get_form_adjusted_baselines(baselines: Dict[str, Dict[str, float]],
                                 season: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Blend the matchup_simulator's raw baselines with form-weighted adjustments.

    The form adjustment is computed as the DELTA between the form-weighted net rating
    and the season net rating, then capped at ±6 points and split 50/50 between
    ortg and drtg.  This keeps ratings in a realistic range while still capturing
    hot/cold streaks.

    Design rationale:
      - Raw baselines already use ROLL20 ratings from the same parquet, so we
        cannot naively add the full Δortg/Δdrtg (double-counts rolling data).
      - Instead we use form_net vs season_net as a pure form signal, then apply
        a capped fractional nudge to base_ortg / base_drtg.

    Parameters
    ----------
    baselines : dict
        Output of matchup_simulator.load_team_baseline_ratings()
    season : str, optional
        NBA season string.

    Returns
    -------
    dict  — same structure as baselines but with form-adjusted values
    """
    # Maximum adjustment to ortg or drtg in either direction (points)
    MAX_FORM_ADJ = 6.0
    # What fraction of the form signal to apply (dampening factor)
    FORM_DAMPENING = 0.5

    form = compute_team_form(season=season)
    adjusted = {}

    for abbrev, base in baselines.items():
        if abbrev not in form:
            logger.warning(f"No form data for {abbrev}, using raw baseline")
            adjusted[abbrev] = base.copy()
            continue

        f = form[abbrev]

        # Form signal: how much better/worse is the team right now vs season average?
        form_delta = f["form_net"] - f["season_net"]   # + = improving, - = declining

        # Apply dampened, capped adjustment
        # Hot team (form_delta > 0) → higher ortg, lower drtg
        # Cold team (form_delta < 0) → lower ortg, higher drtg
        raw_adj = form_delta * FORM_DAMPENING
        capped_adj = float(np.clip(raw_adj, -MAX_FORM_ADJ, MAX_FORM_ADJ))

        # Split the net adjustment half to offense, half to defense
        ortg_nudge = capped_adj * 0.5
        drtg_nudge = -capped_adj * 0.5   # better form → lower drtg (better defense too)

        adj_ortg = base["base_ortg"] + ortg_nudge
        adj_drtg = base["base_drtg"] + drtg_nudge

        adjusted[abbrev] = {
            **base,
            "base_ortg":        round(adj_ortg, 2),
            "base_drtg":        round(adj_drtg, 2),
            "form_meta":        f,
            "form_delta":       round(form_delta, 2),
            "form_ortg_nudge":  round(ortg_nudge, 2),
            "form_drtg_nudge":  round(drtg_nudge, 2),
        }

    return adjusted


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    form = compute_team_form()
    for team in sorted(form.keys()):
        d = form[team]
        print(f"{team}  {d['tier']}  form_net={d['form_net']:+.1f}  "
              f"L5={d['roll5_net']:+.1f}  L10={d['roll10_net']:+.1f}  "
              f"Szn={d['season_net']:+.1f}  "
              f"Δortg={d['ortg_adjustment']:+.1f}  Δdrtg={d['drtg_adjustment']:+.1f}")
