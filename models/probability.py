"""
models/probability.py
─────────────────────
Version 1 NBA Player Performance Probability Engine.

Converts continuous point predictions (plus predicted playing time and context)
into calibrated discrete outcome distributions and over/under/push probabilities
for arbitrary user-supplied thresholds.

Probability estimation methods supported & benchmarked:
  1. Method A — Normal Distribution (continuity-corrected, non-negative discrete CDF)
  2. Method B — Global Residual Bootstrap (10,000 draws from stat OOS residuals)
  3. Method C — Conditional Residual Bootstrap (3-tier comparable context lookup)

Scope restrictions enforced:
  - Estimates player performance & threshold probabilities ONLY.
  - Zero bookmaker odds, fair odds, EV, or betting recommendation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from models.calibration import ProbabilityCalibrator, clip_probs
from models.residuals import canonical_stat_name


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ConditionalConfig:
    """
    Configurable neighbourhood & fallback parameters for Conditional Residual Bootstrap.
    Bandwidths are expressed in normalized distribution-scale units (not hardcoded raw values)
    so the same configuration adapts dynamically across PTS, REB, AST, 3PM, STL, BLK, TOV.
    """
    pred_bandwidth_scale: float = 0.35
    min_bandwidth_scale: float = 0.55
    tier2_pred_bandwidth_scale: float = 0.80
    tier2_min_bandwidth_scale: float = 1.15
    min_samples_tier1: int = 250
    min_samples_tier2: int = 150
    max_knn_z_dist: float = 3.0
    match_starter_role: bool = True
    n_simulations: int = 10_000
    random_seed: int = 42


# ── Discrete Threshold & Line Mathematics ─────────────────────────────────────

def is_integer_line(line: float, tol: float = 1e-9) -> bool:
    """Return True if user-supplied line is effectively an integer (supports pushes)."""
    return abs(line - round(line)) < tol


def _line_boundaries(line: float) -> Tuple[int, int, bool]:
    """
    Map any real-valued threshold `line` onto discrete whole-number boundaries:
      - Over boundary  (k_over): smallest integer strictly > line
      - Under boundary (k_under): largest integer strictly < line (may be -1 if line <= 0)
      - is_int: whether exact push (X == line) is mathematically possible
    """
    if is_integer_line(line):
        k = int(round(line))
        return k + 1, k - 1, True
    return math.floor(line) + 1, math.ceil(line) - 1, False


def calculate_line_probabilities(
    outcomes: np.ndarray,
    line: float,
) -> Tuple[float, float, float]:
    """
    Calculate (prob_over, prob_under, prob_push) from simulated whole-number outcomes
    for any arbitrary user-supplied line (half-point, integer, zero, or fractional).

    Guarantees exact discrete logic and sum(over, under, push) == 1.0.
    """
    arr = np.asarray(outcomes)
    if len(arr) == 0:
        raise ValueError("Cannot calculate line probabilities from empty outcomes array.")

    k_over, k_under, is_int = _line_boundaries(float(line))

    prob_over = float(np.mean(arr >= k_over))
    prob_under = float(np.mean(arr <= k_under)) if k_under >= 0 else 0.0
    prob_push = float(np.mean(arr == int(round(line)))) if is_int and round(line) >= 0 else 0.0

    # Normalise floating-point rounding drift cleanly so sum == 1.0
    total = prob_over + prob_under + prob_push
    if total > 0:
        prob_over /= total
        prob_under /= total
        prob_push /= total

    return prob_over, prob_under, prob_push


def calculate_over_under_probability(
    outcomes: np.ndarray,
    line: float,
) -> Dict[str, float]:
    """Convenience dict returning over, under, and push probabilities."""
    p_over, p_under, p_push = calculate_line_probabilities(outcomes, line)
    return {
        "prob_over": p_over,
        "prob_under": p_under,
        "prob_push": p_push,
    }


# ── Output Data Structures ────────────────────────────────────────────────────

@dataclass
class LineProbabilityResult:
    """Full structured probability result for a single threshold query."""
    stat: str
    line: float
    is_integer_line: bool
    predicted_value: float
    expected_value: float
    distribution_std: float

    # Active probabilities (calibrated if calibrator active, else equal to raw)
    prob_over: float
    prob_under: float
    prob_push: float

    # Preserved raw probabilities
    raw_prob_over: float
    raw_prob_under: float
    raw_prob_push: float

    # Reliability & metadata
    method_used: str
    n_simulations: int
    n_support: int
    fallback_tier: str
    used_fallback: bool
    calibration_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stat": self.stat,
            "line": self.line,
            "is_integer_line": self.is_integer_line,
            "predicted_value": round(self.predicted_value, 2),
            "expected_value": round(self.expected_value, 2),
            "distribution_std": round(self.distribution_std, 2),
            "prob_over": round(self.prob_over, 4),
            "prob_under": round(self.prob_under, 4),
            "prob_push": round(self.prob_push, 4),
            "raw_prob_over": round(self.raw_prob_over, 4),
            "raw_prob_under": round(self.raw_prob_under, 4),
            "raw_prob_push": round(self.raw_prob_push, 4),
            "method_used": self.method_used,
            "n_simulations": self.n_simulations,
            "n_support": self.n_support,
            "fallback_tier": self.fallback_tier,
            "used_fallback": self.used_fallback,
            "calibration_status": self.calibration_status,
        }


@dataclass
class OutcomeDistribution:
    """
    Reusable simulated outcome distribution for one player/game/stat.
    Allows evaluating arbitrarily many lines without regenerating simulations.
    """
    stat: str
    predicted_value: float
    predicted_minutes: float
    outcomes: np.ndarray
    method_used: str
    n_simulations: int
    n_support: int
    fallback_tier: str
    used_fallback: bool
    calibrator: Optional[ProbabilityCalibrator] = None

    # Optional analytical Normal parameters when method_used == 'normal'
    _normal_mu: Optional[float] = None
    _normal_sigma: Optional[float] = None

    @property
    def distribution_mean(self) -> float:
        return float(np.mean(self.outcomes))

    @property
    def distribution_median(self) -> float:
        return float(np.median(self.outcomes))

    @property
    def distribution_std(self) -> float:
        return float(np.std(self.outcomes, ddof=1)) if len(self.outcomes) > 1 else 0.0

    def interval(self, confidence: float = 0.90) -> Tuple[float, float]:
        """Compute empirical central prediction interval (e.g. 50%, 80%, 90%)."""
        alpha = (1.0 - confidence) / 2.0
        lo = float(np.quantile(self.outcomes, alpha))
        hi = float(np.quantile(self.outcomes, 1.0 - alpha))
        return lo, hi

    def _raw_probs_for_line(self, line: float) -> Tuple[float, float, float]:
        if self.method_used == "normal" and self._normal_mu is not None and self._normal_sigma is not None:
            return _normal_cdf_line_probs(self._normal_mu, self._normal_sigma, line)
        return calculate_line_probabilities(self.outcomes, line)

    def _raw_survival_at_threshold(self, k: int) -> float:
        """Raw P(X > k) for integer k."""
        if self.method_used == "normal" and self._normal_mu is not None and self._normal_sigma is not None:
            p_over, _, _ = _normal_cdf_line_probs(self._normal_mu, self._normal_sigma, k + 0.5)
            return p_over
        return float(np.mean(self.outcomes > k))

    def probability_for_line(self, line: float) -> LineProbabilityResult:
        """Compute raw and calibrated Over/Under/Push probabilities for any user line."""
        raw_over, raw_under, raw_push = self._raw_probs_for_line(line)
        is_int = is_integer_line(line)

        if self.calibrator is not None and self.calibrator.method != "none":
            cal_status = self.calibrator.status
            if not is_int:
                cal_over = float(self.calibrator.transform(raw_over))
                cal_under = 1.0 - cal_over
                cal_push = 0.0
            else:
                k = int(round(line))
                # Calibrate monotonic survival probabilities S(k) = P(X > k) and S(k-1) = P(X >= k)
                cal_over = float(self.calibrator.transform(raw_over))
                if k <= 0:
                    cal_under = 0.0
                else:
                    raw_ge = self._raw_survival_at_threshold(k - 1)
                    cal_ge = float(self.calibrator.transform(raw_ge))
                    cal_ge = max(cal_over, cal_ge)  # enforce CDF monotonicity
                    cal_under = max(0.0, 1.0 - cal_ge)
                if cal_over + cal_under > 1.0:
                    s = cal_over + cal_under
                    cal_over /= s
                    cal_under /= s
                cal_push = max(0.0, 1.0 - cal_over - cal_under)
        else:
            cal_status = self.calibrator.status if self.calibrator is not None else "uncalibrated"
            cal_over, cal_under, cal_push = raw_over, raw_under, raw_push

        return LineProbabilityResult(
            stat=self.stat,
            line=float(line),
            is_integer_line=is_int,
            predicted_value=float(self.predicted_value),
            expected_value=self.distribution_mean,
            distribution_std=self.distribution_std,
            prob_over=cal_over,
            prob_under=cal_under,
            prob_push=cal_push,
            raw_prob_over=raw_over,
            raw_prob_under=raw_under,
            raw_prob_push=raw_push,
            method_used=self.method_used,
            n_simulations=self.n_simulations,
            n_support=self.n_support,
            fallback_tier=self.fallback_tier,
            used_fallback=self.used_fallback,
            calibration_status=cal_status,
        )

    def probabilities_for_lines(self, lines: List[float]) -> List[LineProbabilityResult]:
        """Evaluate multiple threshold lines using the same simulated outcome distribution."""
        return [self.probability_for_line(line) for line in lines]


# ── Method A: Normal Distribution ─────────────────────────────────────────────

def _normal_cdf_line_probs(mu: float, sigma: float, line: float) -> Tuple[float, float, float]:
    """
    Exact discrete whole-number non-negative Normal probabilities via continuity correction.
    Equivalent to infinite draws of X = max(0, round(Normal(mu, sigma^2))).
    """
    k_over, k_under, is_int = _line_boundaries(float(line))

    if sigma <= 1e-9:
        val = max(0, int(round(mu)))
        if is_int:
            k = int(round(line))
            return (1.0 if val > k else 0.0, 1.0 if val < k else 0.0, 1.0 if val == k else 0.0)
        return (1.0 if val > line else 0.0, 1.0 if val < line else 0.0, 0.0)

    # Since k_over >= 1 whenever line >= 0, round(Z) >= k_over <==> Z >= k_over - 0.5
    if k_over <= 0:
        p_over = 1.0
    else:
        p_over = float(stats.norm.sf(k_over - 0.5, loc=mu, scale=sigma))

    if k_under < 0:
        p_under = 0.0
    else:
        p_under = float(stats.norm.cdf(k_under + 0.5, loc=mu, scale=sigma))

    p_push = max(0.0, 1.0 - p_over - p_under) if is_int and round(line) >= 0 else 0.0
    total = p_over + p_under + p_push
    return p_over / total, p_under / total, p_push / total


def simulate_normal_outcomes(
    predicted_value: float,
    residuals: np.ndarray,
    n_simulations: int = 10_000,
    seed: int = 42,
    adjust_bias: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Generate whole-number non-negative simulated outcomes under Normal(mu, sigma^2)."""
    res = np.asarray(residuals, dtype=float)
    if len(res) < 2:
        raise ValueError("Need at least 2 historical residuals for Normal estimation.")

    bias = float(np.mean(res)) if adjust_bias else 0.0
    mu = max(0.0, float(predicted_value) + bias)
    sigma = max(float(np.std(res, ddof=1)), 1e-4)

    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=mu, scale=sigma, size=n_simulations)
    outcomes = np.maximum(0, np.rint(raw)).astype(int)
    return outcomes, mu, sigma


def estimate_normal_probability(
    predicted_value: float,
    residuals: np.ndarray,
    line: float,
    n_simulations: int = 10_000,
    seed: int = 42,
    adjust_bias: bool = True,
) -> Dict[str, Any]:
    """Estimate Over/Under/Push probability using Method A (discrete non-negative Normal)."""
    outcomes, mu, sigma = simulate_normal_outcomes(
        predicted_value=predicted_value,
        residuals=residuals,
        n_simulations=n_simulations,
        seed=seed,
        adjust_bias=adjust_bias,
    )
    p_over, p_under, p_push = _normal_cdf_line_probs(mu, sigma, line)
    return {
        "prob_over": p_over,
        "prob_under": p_under,
        "prob_push": p_push,
        "mu": mu,
        "sigma": sigma,
        "n_support": int(len(residuals)),
        "outcomes": outcomes,
    }


# ── Method B: Global Residual Bootstrap ───────────────────────────────────────

def simulate_global_bootstrap_outcomes(
    predicted_value: float,
    residuals: np.ndarray,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate `n_simulations` discrete non-negative outcomes by resampling historical
    out-of-sample residuals uniformly with replacement and adding to `predicted_value`.
    Because raw out-of-sample residuals are `actual - predicted`, resampling them
    automatically preserves systematic bias adjustment.
    """
    res = np.asarray(residuals, dtype=float)
    if len(res) == 0:
        raise ValueError("Cannot simulate outcomes from empty residuals array.")

    rng = np.random.default_rng(seed)
    sampled_res = rng.choice(res, size=n_simulations, replace=True)
    raw = float(predicted_value) + sampled_res
    outcomes = np.maximum(0, np.rint(raw)).astype(int)
    return outcomes


def estimate_global_bootstrap_probability(
    predicted_value: float,
    residuals: np.ndarray,
    line: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Estimate Over/Under/Push probability using Method B (Global Residual Bootstrap)."""
    outcomes = simulate_global_bootstrap_outcomes(
        predicted_value=predicted_value,
        residuals=residuals,
        n_simulations=n_simulations,
        seed=seed,
    )
    p_over, p_under, p_push = calculate_line_probabilities(outcomes, line)
    return {
        "prob_over": p_over,
        "prob_under": p_under,
        "prob_push": p_push,
        "n_support": int(len(residuals)),
        "outcomes": outcomes,
    }


# ── Method C: Conditional Residual Bootstrap ──────────────────────────────────

class ResidualIndex:
    """
    Indexed historical out-of-sample residual store for fast, context-aware
    neighbourhood lookup with explicit hierarchical fallbacks.
    """

    def __init__(
        self,
        stat: str,
        residuals: np.ndarray,
        predicted_values: np.ndarray,
        predicted_minutes: np.ndarray,
        is_starter: Optional[np.ndarray] = None,
        config: Optional[ConditionalConfig] = None,
    ):
        self.stat = canonical_stat_name(stat)
        raw_res = np.asarray(residuals, dtype=float)
        raw_pred = np.asarray(predicted_values, dtype=float)
        raw_min = np.asarray(predicted_minutes, dtype=float)
        self.config = config or ConditionalConfig()

        self.n_total = len(raw_res)
        if self.n_total == 0:
            raise ValueError(f"ResidualIndex for '{self.stat}' initialized with 0 observations.")

        # Sort once by predicted_values so Tier 1 / Tier 2a box queries execute in O(log N) via binary search
        # without allocating giant query_ball_point Python index lists on dense zero-inflated clusters.
        order = np.argsort(raw_pred, kind="mergesort")
        self.residuals = raw_res[order]
        self.predicted_values = raw_pred[order]
        self.predicted_minutes = raw_min[order]

        if is_starter is not None:
            self.is_starter: Optional[np.ndarray] = np.asarray(is_starter, dtype=bool)[order]
        else:
            self.is_starter = None

        # Dynamic distribution scale factors (zero hardcoded stat-specific bounds)
        self.pred_scale = max(float(np.std(self.predicted_values)), 0.1)
        self.min_scale = max(float(np.std(self.predicted_minutes)), 1.0)

        # Normalized 2D coordinate matrix ONLY for bounded Tier 2b fixed-k k-NN fallback
        self._coords_z = np.column_stack([
            self.predicted_values / self.pred_scale,
            self.predicted_minutes / self.min_scale,
        ])
        self._tree = cKDTree(self._coords_z)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        stat: str,
        config: Optional[ConditionalConfig] = None,
    ) -> "ResidualIndex":
        if "ROLL5_STARTER_PCT" in df.columns and df["ROLL5_STARTER_PCT"].notna().any():
            starter_arr = (df["ROLL5_STARTER_PCT"].fillna(0.0) >= 0.5).to_numpy()
        elif "ROLL5_MIN_MEAN" in df.columns:
            starter_arr = (df["ROLL5_MIN_MEAN"].fillna(0.0) >= 24.0).to_numpy()
        else:
            starter_arr = None

        return cls(
            stat=stat,
            residuals=df["residual"].values,
            predicted_values=df["predicted_value"].values,
            predicted_minutes=df["predicted_minutes"].values,
            is_starter=starter_arr,
            config=config,
        )

    def _select_indices(
        self,
        predicted_value: float,
        predicted_minutes: float,
        is_starter: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], str, bool]:
        """
        Internal sorted-slice lookup returning `(indices_or_None_for_all, tier_label, used_fallback)`.
        Returning `None` on Tier 3 avoids copying `np.arange(n_total)` unless required.
        """
        cfg = self.config

        # Tier 1: Closely comparable predictions via O(log N) binary search slice
        t1_pred_tol = cfg.pred_bandwidth_scale * self.pred_scale
        t1_min_tol = cfg.min_bandwidth_scale * self.min_scale

        lo1 = int(np.searchsorted(self.predicted_values, predicted_value - t1_pred_tol, side="left"))
        hi1 = int(np.searchsorted(self.predicted_values, predicted_value + t1_pred_tol, side="right"))

        if hi1 - lo1 >= cfg.min_samples_tier1:
            m1 = np.abs(self.predicted_minutes[lo1:hi1] - predicted_minutes) <= t1_min_tol
            if cfg.match_starter_role and is_starter is not None and self.is_starter is not None:
                m1 &= (self.is_starter[lo1:hi1] == bool(is_starter))
            local1 = np.flatnonzero(m1)
            if len(local1) >= cfg.min_samples_tier1:
                return lo1 + local1, "tier1_tight_comparable", False

        # Tier 2a: Broad comparable box (drops strict starter flag & widens tolerances)
        t2_pred_tol = cfg.tier2_pred_bandwidth_scale * self.pred_scale
        t2_min_tol = cfg.tier2_min_bandwidth_scale * self.min_scale

        lo2 = int(np.searchsorted(self.predicted_values, predicted_value - t2_pred_tol, side="left"))
        hi2 = int(np.searchsorted(self.predicted_values, predicted_value + t2_pred_tol, side="right"))

        if hi2 - lo2 >= cfg.min_samples_tier2:
            m2 = np.abs(self.predicted_minutes[lo2:hi2] - predicted_minutes) <= t2_min_tol
            local2 = np.flatnonzero(m2)
            if len(local2) >= cfg.min_samples_tier2:
                return lo2 + local2, "tier2_broad_comparable", True

        # Tier 2b: Bounded fixed-k k-NN in standardized (pred, min) space for high-volume tail stars
        if self.n_total >= cfg.min_samples_tier2:
            q_z = np.array([predicted_value / self.pred_scale, predicted_minutes / self.min_scale])
            dists, nn_idx = self._tree.query(q_z, k=cfg.min_samples_tier2)
            nn_arr = np.atleast_1d(np.asarray(nn_idx, dtype=int))
            if float(np.max(dists)) <= cfg.max_knn_z_dist:
                return nn_arr, "tier2_knn_comparable", True

        # Tier 3: Global fallback (used when entire pool is small or query is an extreme outlier)
        return None, "tier3_global_fallback", True

    def select_residuals(
        self,
        predicted_value: float,
        predicted_minutes: float,
        is_starter: Optional[bool] = None,
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Retrieve comparable historical residuals for `(predicted_value, predicted_minutes)`
        following a strict 3-tier fallback hierarchy:
          1. tier1_tight_comparable  : tight pred & minutes box (+ starter alignment if available)
          2. tier2_broad_comparable  : broader pred & minutes box, or bounded k-NN
          3. tier3_global_fallback   : entire historical residual pool for the stat
        """
        idx, tier, used_fb = self._select_indices(
            predicted_value=predicted_value,
            predicted_minutes=predicted_minutes,
            is_starter=is_starter,
        )
        if idx is None:
            return self.residuals, tier, used_fb
        return self.residuals[idx], tier, used_fb

    def batch_select_indices(
        self,
        pred_values: np.ndarray,
        pred_minutes: np.ndarray,
        is_starter: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Fast memory-safe batch index selection using O(log N) binary search slices.
        Returns list of index arrays and fallback tier labels for each query row.
        """
        n_q = len(pred_values)
        all_indices: List[np.ndarray] = []
        tiers: List[str] = []
        global_idx = np.arange(self.n_total)
        has_starter = is_starter is not None and self.is_starter is not None

        for i in range(n_q):
            st_i = bool(is_starter[i]) if has_starter else None
            idx, tier, _ = self._select_indices(float(pred_values[i]), float(pred_minutes[i]), is_starter=st_i)
            all_indices.append(global_idx if idx is None else idx)
            tiers.append(tier)

        return all_indices, tiers


def simulate_conditional_bootstrap_outcomes(
    predicted_value: float,
    predicted_minutes: float,
    residual_index: ResidualIndex,
    is_starter: Optional[bool] = None,
    n_simulations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int, str, bool]:
    """
    Simulate discrete non-negative outcomes using Method C (Conditional Residual Bootstrap).
    """
    cfg = residual_index.config
    n_sim = n_simulations or cfg.n_simulations
    rng_seed = cfg.random_seed if seed is None else seed

    cond_res, tier, used_fallback = residual_index.select_residuals(
        predicted_value=predicted_value,
        predicted_minutes=predicted_minutes,
        is_starter=is_starter,
    )

    rng = np.random.default_rng(rng_seed)
    sampled = rng.choice(cond_res, size=n_sim, replace=True)
    raw = float(predicted_value) + sampled
    outcomes = np.maximum(0, np.rint(raw)).astype(int)

    return outcomes, int(len(cond_res)), tier, used_fallback


def estimate_conditional_bootstrap_probability(
    predicted_value: float,
    predicted_minutes: float,
    residual_index: ResidualIndex,
    line: float,
    is_starter: Optional[bool] = None,
    n_simulations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate Over/Under/Push probability using Method C (Conditional Residual Bootstrap)."""
    outcomes, n_support, tier, used_fallback = simulate_conditional_bootstrap_outcomes(
        predicted_value=predicted_value,
        predicted_minutes=predicted_minutes,
        residual_index=residual_index,
        is_starter=is_starter,
        n_simulations=n_simulations,
        seed=seed,
    )
    p_over, p_under, p_push = calculate_line_probabilities(outcomes, line)
    return {
        "prob_over": p_over,
        "prob_under": p_under,
        "prob_push": p_push,
        "n_support": n_support,
        "fallback_tier": tier,
        "used_fallback": used_fallback,
        "outcomes": outcomes,
    }


# ── Unified Distribution Builder ──────────────────────────────────────────────

def simulate_player_outcomes(
    stat: str,
    predicted_value: float,
    predicted_minutes: float,
    residual_index: ResidualIndex,
    method: str = "conditional_bootstrap",
    is_starter: Optional[bool] = None,
    n_simulations: int = 10_000,
    seed: int = 42,
    calibrator: Optional[ProbabilityCalibrator] = None,
) -> OutcomeDistribution:
    """
    Build a reusable `OutcomeDistribution` for one player/stat prediction.
    Supports 'normal', 'global_bootstrap', and 'conditional_bootstrap'.
    """
    canon = canonical_stat_name(stat)
    m_name = method.strip().lower()
    pred_clamped = max(0.0, float(predicted_value))

    if m_name == "normal":
        outcomes, mu, sigma = simulate_normal_outcomes(
            predicted_value=pred_clamped,
            residuals=residual_index.residuals,
            n_simulations=n_simulations,
            seed=seed,
            adjust_bias=True,
        )
        return OutcomeDistribution(
            stat=canon,
            predicted_value=pred_clamped,
            predicted_minutes=float(predicted_minutes),
            outcomes=outcomes,
            method_used="normal",
            n_simulations=n_simulations,
            n_support=residual_index.n_total,
            fallback_tier="global_parametric",
            used_fallback=False,
            calibrator=calibrator,
            _normal_mu=mu,
            _normal_sigma=sigma,
        )

    if m_name == "global_bootstrap":
        outcomes = simulate_global_bootstrap_outcomes(
            predicted_value=pred_clamped,
            residuals=residual_index.residuals,
            n_simulations=n_simulations,
            seed=seed,
        )
        return OutcomeDistribution(
            stat=canon,
            predicted_value=pred_clamped,
            predicted_minutes=float(predicted_minutes),
            outcomes=outcomes,
            method_used="global_bootstrap",
            n_simulations=n_simulations,
            n_support=residual_index.n_total,
            fallback_tier="global_empirical",
            used_fallback=False,
            calibrator=calibrator,
        )

    if m_name == "conditional_bootstrap":
        outcomes, n_support, tier, used_fb = simulate_conditional_bootstrap_outcomes(
            predicted_value=pred_clamped,
            predicted_minutes=predicted_minutes,
            residual_index=residual_index,
            is_starter=is_starter,
            n_simulations=n_simulations,
            seed=seed,
        )
        return OutcomeDistribution(
            stat=canon,
            predicted_value=pred_clamped,
            predicted_minutes=float(predicted_minutes),
            outcomes=outcomes,
            method_used="conditional_bootstrap",
            n_simulations=n_simulations,
            n_support=n_support,
            fallback_tier=tier,
            used_fallback=used_fb,
            calibrator=calibrator,
        )

    raise ValueError(f"Unknown probability method '{method}'. Choose normal, global_bootstrap, or conditional_bootstrap.")


# ══════════════════════════════════════════════════════════════════════════════
# Backward-Compatible Legacy Helpers (Preserved for existing imports/tests)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PropPrediction:
    """Legacy prediction dataclass preserved for backwards compatibility."""
    player_id: int
    player_name: str
    stat: str
    line: float
    prob_over: float
    prob_under: float
    predicted_mean: float
    predicted_std: float
    recent_avg_5: Optional[float] = None
    recent_avg_10: Optional[float] = None
    season_avg: Optional[float] = None
    career_avg: Optional[float] = None
    ci_low: float = 0.0
    ci_high: float = 0.0
    n_games_history: int = 0
    method: str = "parametric"
    model_name: str = ""
    opponent_abbrev: str = ""
    is_home: int = 1
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


def prob_over_parametric(
    predicted_mean: float,
    predicted_std: float,
    line: float,
    clip: float = 1e-4,
) -> float:
    """Legacy parametric helper."""
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
    """Legacy empirical helper."""
    if len(history) < 5:
        return prob_over_parametric(predicted_mean, predicted_std, line)
    rng = np.random.default_rng(random_seed)
    history = np.asarray(history, dtype=float)
    hist_mean = np.mean(history)
    hist_std = np.std(history)
    if hist_std < 0.5:
        return prob_over_parametric(predicted_mean, predicted_std, line)
    standardised = (history - hist_mean) / hist_std
    simulated = standardised * predicted_std + predicted_mean
    samples = rng.choice(simulated, size=n_bootstrap, replace=True)
    noise = rng.normal(0, predicted_std * 0.1, size=n_bootstrap)
    samples = np.clip(samples + noise, 0, None)
    prob = float(np.mean(samples > line))
    return float(np.clip(prob, 1e-4, 1.0 - 1e-4))


def compute_confidence_interval(
    predicted_mean: float,
    predicted_std: float,
    confidence: float = 0.90,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    return (max(0.0, predicted_mean - z * predicted_std), predicted_mean + z * predicted_std)


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
