"""
models/baseline.py
──────────────────
Simple baseline predictors. Every ML model must beat these to be useful.

Baselines implemented:
  1. SeasonAverageBaseline  — predicts the player's season-to-date average
  2. RecentAverageBaseline  — predicts the player's last-N-games average
  3. CareerAverageBaseline  — predicts the player's career average
  4. NaiveTeamWinBaseline   — predicts win for the home team (home-court bias)

All baselines expose a common interface:
  .predict(X)         → np.ndarray of predicted stat values
  .predict_proba(X)   → np.ndarray of [P(under), P(over)] for a given line
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class SeasonAverageBaseline:
    """
    Predict the player's season-to-date average as the expected value.
    Uses 'SEASON_{stat}_MEAN' column from the feature matrix.
    """

    def __init__(self, stat: str):
        self.stat = stat
        self.season_col = f"SEASON_{stat}_MEAN"
        self.std_: float = 5.0  # fallback std, updated by fit()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonAverageBaseline":
        """Estimate residual std from training data."""
        preds = self.predict(X)
        valid = ~np.isnan(preds)
        if valid.sum() > 10:
            residuals = y.values[valid] - preds[valid]
            self.std_ = float(np.std(residuals))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.season_col in X.columns:
            return X[self.season_col].fillna(X[self.season_col].median()).values
        # Fallback: use any available rolling mean
        for w in [10, 20, 5]:
            col = f"ROLL{w}_{self.stat}_MEAN"
            if col in X.columns:
                return X[col].fillna(X[col].median()).values
        return np.full(len(X), np.nan)

    def predict_proba_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """Return P(stat > line) using normal distribution around season average."""
        means = self.predict(X)
        return stats.norm.sf(line, loc=means, scale=self.std_)

    def __repr__(self) -> str:
        return f"SeasonAverageBaseline(stat={self.stat}, std={self.std_:.2f})"


class RecentAverageBaseline:
    """
    Predict using the player's last-N-games rolling average.
    Uses 'ROLL{window}_{stat}_MEAN' column.
    """

    def __init__(self, stat: str, window: int = 10):
        self.stat = stat
        self.window = window
        self.roll_col = f"ROLL{window}_{stat}_MEAN"
        self.std_: float = 5.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecentAverageBaseline":
        preds = self.predict(X)
        valid = ~np.isnan(preds)
        if valid.sum() > 10:
            residuals = y.values[valid] - preds[valid]
            self.std_ = float(np.std(residuals))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.roll_col in X.columns:
            return X[self.roll_col].fillna(X[self.roll_col].median()).values
        return np.full(len(X), np.nan)

    def predict_proba_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        means = self.predict(X)
        return stats.norm.sf(line, loc=means, scale=self.std_)

    def __repr__(self) -> str:
        return f"RecentAverageBaseline(stat={self.stat}, window={self.window}, std={self.std_:.2f})"


class CareerAverageBaseline:
    """Predict using career-to-date average."""

    def __init__(self, stat: str):
        self.stat = stat
        self.career_col = f"CAREER_{stat}_MEAN"
        self.std_: float = 5.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CareerAverageBaseline":
        preds = self.predict(X)
        valid = ~np.isnan(preds)
        if valid.sum() > 10:
            residuals = y.values[valid] - preds[valid]
            self.std_ = float(np.std(residuals))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.career_col in X.columns:
            return X[self.career_col].fillna(X[self.career_col].median()).values
        return np.full(len(X), np.nan)

    def predict_proba_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        means = self.predict(X)
        return stats.norm.sf(line, loc=means, scale=self.std_)


class NaiveHomeWinBaseline:
    """
    Naive team win baseline: always predict home team wins.
    Historical NBA home-win rate is ~58%.
    """

    HOME_WIN_RATE = 0.578

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveHomeWinBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(
            X["IS_HOME"] == 1,
            self.HOME_WIN_RATE,
            1.0 - self.HOME_WIN_RATE,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict(X)
        return np.column_stack([1.0 - probs, probs])


class TeamNetRatingBaseline:
    """
    Team win baseline using rolling net rating differential.
    Home team wins if (home_net_rating - away_net_rating) > threshold.
    """

    def __init__(self, window: int = 10, home_advantage: float = 3.0):
        self.window = window
        self.home_advantage = home_advantage
        self.logistic_scale_: float = 10.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TeamNetRatingBaseline":
        """Calibrate the logistic scale on training data."""
        from scipy.optimize import minimize_scalar
        diffs = self._get_net_rating_diff(X)

        def neg_log_likelihood(scale):
            probs = 1 / (1 + np.exp(-diffs / scale))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y.values * np.log(probs) + (1 - y.values) * np.log(1 - probs))

        result = minimize_scalar(neg_log_likelihood, bounds=(1, 50), method="bounded")
        self.logistic_scale_ = float(result.x)
        return self

    def _get_net_rating_diff(self, X: pd.DataFrame) -> np.ndarray:
        col = f"ROLL{self.window}_NET_RATING"
        home_adj = self.home_advantage * (2 * X["IS_HOME"].values - 1)
        if col in X.columns:
            return X[col].fillna(0).values + home_adj
        return home_adj

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        diff = self._get_net_rating_diff(X)
        prob_win = 1 / (1 + np.exp(-diff / self.logistic_scale_))
        return np.column_stack([1 - prob_win, prob_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1]
