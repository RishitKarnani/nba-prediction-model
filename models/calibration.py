"""
models/calibration.py
─────────────────────
Probability calibration utilities (Platt scaling & Isotonic regression) and
proper scoring rules (Brier Score, Log Loss, Expected Calibration Error, Reliability curve).

Design rules (Phase 5):
- Evaluates Platt scaling vs Isotonic regression strictly on a chronological
  out-of-sample calibration split — never on the final test holdout.
- Never assumes calibration helps: selects 'none' (raw probabilities) unless
  calibration improves out-of-sample validation Brier Score and Log Loss.
- Guards against small calibration sample sizes (returns raw if insufficient data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


MIN_PROB_CLIP = 1e-4
MAX_PROB_CLIP = 1.0 - 1e-4


def clip_probs(p: np.ndarray | float) -> np.ndarray | float:
    """Clip probabilities away from exact 0/1 for numerical stability in log-loss/logits."""
    return np.clip(p, MIN_PROB_CLIP, MAX_PROB_CLIP)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score (mean squared probability error). Lower is better."""
    return float(brier_score_loss(np.asarray(y_true, dtype=int), clip_probs(np.asarray(y_prob, dtype=float))))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log Loss (binary cross-entropy). Lower is better."""
    probs = clip_probs(np.asarray(y_prob, dtype=float))
    return float(log_loss(np.asarray(y_true, dtype=int), probs, labels=[0, 1]))


def compute_reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Compute reliability diagram buckets and Expected Calibration Error (ECE).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_info: List[Dict[str, Any]] = []
    ece = 0.0
    n_total = len(y_t)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_p >= lo) & (y_p <= hi)
        else:
            mask = (y_p >= lo) & (y_p < hi)

        cnt = int(mask.sum())
        if cnt > 0:
            mean_pred = float(np.mean(y_p[mask]))
            actual_freq = float(np.mean(y_t[mask]))
            gap = abs(actual_freq - mean_pred)
            ece += (cnt / max(1, n_total)) * gap
            bins_info.append({
                "bin": f"[{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'}",
                "count": cnt,
                "mean_predicted_prob": round(mean_pred, 4),
                "actual_frequency": round(actual_freq, 4),
                "abs_gap": round(gap, 4),
            })

    return bins_info, round(float(ece), 5)


@dataclass
class ProbabilityCalibrator:
    """
    Calibrator for a single statistic (`method` in {'none', 'platt', 'isotonic'}).
    """
    stat: str
    method: str = "none"
    status: str = "uncalibrated"
    n_calibration_samples: int = 0
    raw_val_brier: Optional[float] = None
    cal_val_brier: Optional[float] = None
    raw_val_logloss: Optional[float] = None
    cal_val_logloss: Optional[float] = None
    _platt_model: Optional[LogisticRegression] = field(default=None, repr=False)
    _iso_model: Optional[IsotonicRegression] = field(default=None, repr=False)

    @staticmethod
    def _to_logits(p: np.ndarray) -> np.ndarray:
        pc = clip_probs(p)
        return np.log(pc / (1.0 - pc)).reshape(-1, 1)

    def fit_and_select(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        min_samples: int = 500,
        train_frac: float = 0.6,
        min_improvement_brier: float = 5e-4,
    ) -> "ProbabilityCalibrator":
        """
        Chronologically split `(p_raw, y_true)` into calibration-fit (first `train_frac`)
        and calibration-eval (remaining `1 - train_frac`), test Platt vs Isotonic vs Raw,
        and if one wins on BOTH Brier Score and Log Loss, refit the winning calibrator
        on the full calibration dataset.
        """
        p_arr = clip_probs(np.asarray(p_raw, dtype=float))
        y_arr = np.asarray(y_true, dtype=int)
        n = len(p_arr)
        self.n_calibration_samples = n

        if n < min_samples or len(np.unique(y_arr)) < 2:
            self.method = "none"
            self.status = "insufficient_calibration_data"
            return self

        split = int(n * train_frac)
        p_fit, y_fit = p_arr[:split], y_arr[:split]
        p_eval, y_eval = p_arr[split:], y_arr[split:]

        if len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
            self.method = "none"
            self.status = "insufficient_class_balance"
            return self

        raw_brier = compute_brier_score(y_eval, p_eval)
        raw_ll = compute_log_loss(y_eval, p_eval)

        # Candidate 1: Platt Scaling (logistic regression on log-odds)
        platt = LogisticRegression(C=1.0, solver="lbfgs")
        platt.fit(self._to_logits(p_fit), y_fit)
        p_platt_eval = clip_probs(platt.predict_proba(self._to_logits(p_eval))[:, 1])
        platt_brier = compute_brier_score(y_eval, p_platt_eval)
        platt_ll = compute_log_loss(y_eval, p_platt_eval)

        # Candidate 2: Isotonic Regression
        iso = IsotonicRegression(y_min=MIN_PROB_CLIP, y_max=MAX_PROB_CLIP, out_of_bounds="clip")
        iso.fit(p_fit, y_fit)
        p_iso_eval = clip_probs(iso.transform(p_eval))
        iso_brier = compute_brier_score(y_eval, p_iso_eval)
        iso_ll = compute_log_loss(y_eval, p_iso_eval)

        # Compare strictly against Raw on BOTH Brier and Log Loss
        candidates = [
            ("none", raw_brier, raw_ll),
        ]
        if (raw_brier - platt_brier) > min_improvement_brier and platt_ll < raw_ll:
            candidates.append(("platt", platt_brier, platt_ll))
        if (raw_brier - iso_brier) > min_improvement_brier and iso_ll < raw_ll:
            candidates.append(("isotonic", iso_brier, iso_ll))

        # Select lowest Log Loss among qualifying improvements (tie-break: Brier)
        best_method, best_brier, best_ll = min(candidates, key=lambda x: (x[2], x[1]))

        self.method = best_method
        self.raw_val_brier = round(raw_brier, 6)
        self.raw_val_logloss = round(raw_ll, 6)
        self.cal_val_brier = round(best_brier, 6)
        self.cal_val_logloss = round(best_ll, 6)

        if best_method == "none":
            self.status = "raw_retained_no_calibration_gain"
            self._platt_model = None
            self._iso_model = None
        elif best_method == "platt":
            final_platt = LogisticRegression(C=1.0, solver="lbfgs")
            final_platt.fit(self._to_logits(p_arr), y_arr)
            self._platt_model = final_platt
            self._iso_model = None
            self.status = "calibrated_platt"
        else:
            final_iso = IsotonicRegression(y_min=MIN_PROB_CLIP, y_max=MAX_PROB_CLIP, out_of_bounds="clip")
            final_iso.fit(p_arr, y_arr)
            self._iso_model = final_iso
            self._platt_model = None
            self.status = "calibrated_isotonic"

        return self

    def transform(self, p_raw: np.ndarray | float) -> np.ndarray | float:
        """Transform raw probability or array of probabilities using fitted calibrator."""
        is_scalar = np.isscalar(p_raw)
        arr = clip_probs(np.atleast_1d(np.asarray(p_raw, dtype=float)))

        if self.method == "platt" and self._platt_model is not None:
            out = clip_probs(self._platt_model.predict_proba(self._to_logits(arr))[:, 1])
        elif self.method == "isotonic" and self._iso_model is not None:
            out = clip_probs(self._iso_model.transform(arr))
        else:
            out = arr

        return float(out[0]) if is_scalar else out

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "stat": self.stat,
            "method": self.method,
            "status": self.status,
            "n_calibration_samples": self.n_calibration_samples,
            "raw_val_brier": self.raw_val_brier,
            "cal_val_brier": self.cal_val_brier,
            "raw_val_logloss": self.raw_val_logloss,
            "cal_val_logloss": self.cal_val_logloss,
        }

