"""
models/probability.py
─────────────────────
Converts a predicted performance distribution into over/under probabilities.

Design:
  The prediction engine produces:
    - predicted_mean  : expected stat value from the ML model
    - predicted_std   : estimated standard deviation (from residual model or bootstrap)

  This module converts those into:
    - P(stat > line)  : over probability
    - P(stat < line)  : under probability
    - confidence interval
    - fair decimal odds

Two methods are supported:
  1. Parametric (Normal distribution)  — fast, smooth
  2. Empirical (historical game values) — captures non-normal shapes (e.g. skewed distributions for low-minute players)

The empirical method is preferred when sufficient history (≥ 15 games) exists.
The parametric method is used as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats


@dataclass
class PropPrediction:
    """
    Full output object for a single player prop prediction.
    All probabilities are in [0, 1].
    """
    player_id: int
    player_name: str
    stat: str
    line: float

    # Core probabilities
    prob_over: float
    prob_under: float

    # Point estimates
    predicted_mean: float
    predicted_std: float

    # Context
    recent_avg_5: Optional[float] = None
    recent_avg_10: Optional[float] = None
    season_avg: Optional[float] = None
    career_avg: Optional[float] = None

    # Confidence interval (90%)
    ci_low: float = 0.0
    ci_high: float = 0.0

    # Meta
    n_games_history: int = 0
    method: str = "parametric"   # "parametric" or "empirical"
    model_name: str = ""
    opponent_abbrev: str = ""
    is_home: int = 1

    # Raw history used for empirical method
    history_values: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "stat": self.stat,
            "line": self.line,
            "prob_over": round(self.prob_over, 4),
            "prob_under": round(self.prob_under, 4),
            "predicted_mean": round(self.predicted_mean, 2),
            "predicted_std": round(self.predicted_std, 2),
            "recent_avg_5": round(self.recent_avg_5, 2) if self.recent_avg_5 is not None else None,
            "recent_avg_10": round(self.recent_avg_10, 2) if self.recent_avg_10 is not None else None,
            "season_avg": round(self.season_avg, 2) if self.season_avg is not None else None,
            "career_avg": round(self.career_avg, 2) if self.career_avg is not None else None,
            "ci_low": round(self.ci_low, 2),
            "ci_high": round(self.ci_high, 2),
            "n_games_history": self.n_games_history,
            "method": self.method,
            "model_name": self.model_name,
            "opponent_abbrev": self.opponent_abbrev,
            "is_home": self.is_home,
        }


@dataclass
class TeamPrediction:
    """Full output object for a team win probability prediction."""
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str

    prob_home_win: float
    prob_away_win: float

    home_rolling_net_rating: Optional[float] = None
    away_rolling_net_rating: Optional[float] = None
    expected_total: Optional[float] = None
    model_name: str = ""

    def to_dict(self) -> dict:
        return {
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "home_team_name": self.home_team_name,
            "away_team_name": self.away_team_name,
            "prob_home_win": round(self.prob_home_win, 4),
            "prob_away_win": round(self.prob_away_win, 4),
            "home_rolling_net_rating": self.home_rolling_net_rating,
            "away_rolling_net_rating": self.away_rolling_net_rating,
            "expected_total": self.expected_total,
            "model_name": self.model_name,
        }


# ── Probability calculation functions ────────────────────────────────────────

def prob_over_parametric(
    predicted_mean: float,
    predicted_std: float,
    line: float,
    clip: float = 1e-4,
) -> float:
    """
    Compute P(stat > line) assuming a Normal distribution.

    Parameters
    ----------
    predicted_mean : model's expected value for the stat
    predicted_std  : model's estimated standard deviation
    line           : bookmaker's line (e.g. 25.5 for points)
    clip           : minimum probability to avoid 0/1 extremes

    Returns
    -------
    float in (clip, 1-clip)
    """
    if predicted_std <= 0:
        return float(clip if predicted_mean <= line else 1.0 - clip)
    prob = float(stats.norm.sf(line, loc=predicted_mean, scale=predicted_std))
    return float(np.clip(prob, clip, 1.0 - clip))


def prob_over_empirical(
    history: np.ndarray,
    predicted_mean: float,
    predicted_std: float,
    line: float,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> float:
    """
    Compute P(stat > line) using bootstrapped empirical distribution,
    shifted/scaled to the model's predicted mean and std.

    This captures non-normal shapes (e.g. bimodal distributions for players
    who either go OFF or go cold).

    Parameters
    ----------
    history        : array of recent game stat values (e.g. last 20 games)
    predicted_mean : model's predicted mean (used to shift the distribution)
    predicted_std  : model's predicted std  (used to scale the distribution)
    line           : over/under line
    n_bootstrap    : number of bootstrap samples

    Returns
    -------
    float probability P(stat > line)
    """
    if len(history) < 5:
        return prob_over_parametric(predicted_mean, predicted_std, line)

    rng = np.random.default_rng(random_seed)
    history = np.asarray(history, dtype=float)

    # Centre and scale historical distribution
    hist_mean = np.mean(history)
    hist_std = np.std(history)

    if hist_std < 0.5:
        # Too little variance — fall back to parametric
        return prob_over_parametric(predicted_mean, predicted_std, line)

    # Standardise, shift to model prediction
    standardised = (history - hist_mean) / hist_std
    simulated = standardised * predicted_std + predicted_mean

    # Bootstrap: resample simulated values
    samples = rng.choice(simulated, size=n_bootstrap, replace=True)

    # Add small normal noise to smooth the distribution
    noise = rng.normal(0, predicted_std * 0.1, size=n_bootstrap)
    samples = samples + noise

    # Clip to realistic range (can't score negative)
    samples = np.clip(samples, 0, None)

    prob = float(np.mean(samples > line))
    return float(np.clip(prob, 1e-4, 1.0 - 1e-4))


def compute_confidence_interval(
    predicted_mean: float,
    predicted_std: float,
    confidence: float = 0.90,
) -> tuple[float, float]:
    """Return (lower, upper) confidence interval bounds."""
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    return (
        max(0.0, predicted_mean - z * predicted_std),
        predicted_mean + z * predicted_std,
    )


def build_prop_prediction(
    player_id: int,
    player_name: str,
    stat: str,
    line: float,
    predicted_mean: float,
    predicted_std: float,
    history: Optional[np.ndarray] = None,
    recent_avg_5: Optional[float] = None,
    recent_avg_10: Optional[float] = None,
    season_avg: Optional[float] = None,
    career_avg: Optional[float] = None,
    opponent_abbrev: str = "",
    is_home: int = 1,
    model_name: str = "",
    empirical_threshold: int = 15,
) -> PropPrediction:
    """
    Build a complete PropPrediction from model outputs.

    Automatically chooses empirical vs. parametric method.
    """
    use_empirical = history is not None and len(history) >= empirical_threshold

    if use_empirical:
        prob_over = prob_over_empirical(history, predicted_mean, predicted_std, line)
        method = "empirical"
    else:
        prob_over = prob_over_parametric(predicted_mean, predicted_std, line)
        method = "parametric"

    prob_under = 1.0 - prob_over
    ci_low, ci_high = compute_confidence_interval(predicted_mean, predicted_std)

    return PropPrediction(
        player_id=player_id,
        player_name=player_name,
        stat=stat,
        line=line,
        prob_over=prob_over,
        prob_under=prob_under,
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
        recent_avg_5=recent_avg_5,
        recent_avg_10=recent_avg_10,
        season_avg=season_avg,
        career_avg=career_avg,
        ci_low=ci_low,
        ci_high=ci_high,
        n_games_history=len(history) if history is not None else 0,
        method=method,
        model_name=model_name,
        opponent_abbrev=opponent_abbrev,
        is_home=is_home,
        history_values=list(history) if history is not None else [],
    )
