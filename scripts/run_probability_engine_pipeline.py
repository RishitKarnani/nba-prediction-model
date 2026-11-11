"""
Version 1 NBA Player Performance Probability Engine — Full Validation & Calibration Pipeline.

Executes strictly chronological validation without data leakage:
  Phase 1 : Verify/generate genuine out-of-sample predictions & residuals (2022-27)
  Phase 2 : Residual distribution analysis & 5 diagnostic plots (2022-25 OOS history)
  Phase 3/4: Method Comparison (Normal vs Global Bootstrap vs Conditional Bootstrap)
             + ConditionalConfig tuning on 2024-25 walk-forward OOS (using 2022-24 residuals)
  Phase 5 : Chronological calibration selection (Platt vs Isotonic vs Raw) on 2024-25
             -> persists `artefacts/probability/engine_config.json` & `calibrators.joblib`
  Phase 6 : Backtesting on unseen 2025-26 (2022-25 residuals) and holdout 2026-27 (2022-26 residuals)
             + subgroup breakdowns, reliability diagrams, and coverage plots
  Phase 7 : End-to-end PlayerProbabilityEngine matchup verification

Scope restrictions honored:
  - Strictly player performance prediction & threshold probabilities.
  - NO odds, bookmaker lines, fair odds, EV calculations, betting advice, PRA, or injury features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from models.calibration import (
    ProbabilityCalibrator,
    compute_brier_score,
    compute_log_loss,
    compute_reliability_bins,
)
from models.inference import PlayerProbabilityEngine
from models.probability import (
    ConditionalConfig,
    ResidualIndex,
    _normal_cdf_line_probs,
    calculate_line_probabilities,
)
from models.residuals import (
    TARGETS,
    analyse_residual_distribution,
    generate_out_of_sample_predictions,
    load_oos_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("prob_pipeline")

PROB_ARTEFACT_DIR = settings.artefacts_dir / "probability"
PLOT_DIR = settings.artefacts_dir / "plots" / "probability"

# Fixed milestone lines per stat for secondary multi-threshold validation
MILESTONE_LINES: Dict[str, List[float]] = {
    "PTS": [9.5, 14.5, 19.5, 24.5],
    "REB": [3.5, 5.5, 7.5],
    "AST": [1.5, 3.5, 5.5],
    "3PM": [0.5, 1.5, 2.5],
    "STL": [0.5, 1.5],
    "BLK": [0.5, 1.5],
    "TOV": [0.5, 1.5, 2.5],
}

ANCHOR_STEP_BY_STAT: Dict[str, float] = {
    "PTS": 4.0,
    "REB": 2.0,
    "AST": 2.0,
    "3PM": 1.0,
    "STL": 1.0,
    "BLK": 1.0,
    "TOV": 1.0,
}

# Candidate ConditionalConfig presets evaluated strictly on 2024-25 selection split
CONFIG_CANDIDATES: Dict[str, ConditionalConfig] = {
    "tight": ConditionalConfig(
        pred_bandwidth_scale=0.25,
        min_bandwidth_scale=0.40,
        tier2_pred_bandwidth_scale=0.65,
        tier2_min_bandwidth_scale=0.95,
        min_samples_tier1=200,
        min_samples_tier2=120,
        n_simulations=10_000,
        random_seed=42,
    ),
    "balanced": ConditionalConfig(
        pred_bandwidth_scale=0.35,
        min_bandwidth_scale=0.55,
        tier2_pred_bandwidth_scale=0.80,
        tier2_min_bandwidth_scale=1.15,
        min_samples_tier1=250,
        min_samples_tier2=150,
        n_simulations=10_000,
        random_seed=42,
    ),
    "wide": ConditionalConfig(
        pred_bandwidth_scale=0.50,
        min_bandwidth_scale=0.75,
        tier2_pred_bandwidth_scale=1.00,
        tier2_min_bandwidth_scale=1.35,
        min_samples_tier1=300,
        min_samples_tier2=150,
        n_simulations=10_000,
        random_seed=42,
    ),
}


def compute_anchor_lines(df: pd.DataFrame) -> np.ndarray:
    """
    Pre-game rolling player benchmark lines anchored strictly on ROLL10_STAT_MEAN
    (falls back to predicted_value if insufficient history). Always half-point >= 0.5.
    """
    roll_ref = df["ROLL10_STAT_MEAN"].fillna(df["predicted_value"]).to_numpy(dtype=float)
    return np.maximum(0.5, np.floor(roll_ref) + 0.5)


def compute_multi_lines_matrix(df: pd.DataFrame, stat: str) -> np.ndarray:
    """
    Construct a 3-line matrix per row:
      [lower_line, anchor_line, upper_line]
    covering distinct probability regions (~70-80%, ~50%, ~20-30%) without leaking outcomes.
    """
    anchor = compute_anchor_lines(df)
    step = ANCHOR_STEP_BY_STAT[stat]
    lo = np.where(anchor - step >= 0.5, anchor - step, anchor + 2.0 * step)
    hi = anchor + step
    return np.column_stack([lo, anchor, hi])


# ── Batch Simulation Helpers (Fast, Exact Match to models/probability.py) ─────

def evaluate_normal_batch(
    pred: np.ndarray,
    actual: np.ndarray,
    lines_mat: np.ndarray,
    hist_res: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate Method A (Bias-adjusted continuity-corrected discrete Normal) in batch."""
    bias = float(np.mean(hist_res))
    sigma = max(float(np.std(hist_res, ddof=1)), 1e-4)
    mu = np.maximum(0.0, pred + bias)

    n_rows, n_lines = lines_mat.shape
    p_over = np.empty((n_rows, n_lines), dtype=float)
    for j in range(n_lines):
        k_over = np.floor(lines_mat[:, j]) + 1.0
        p_j = stats.norm.sf(k_over - 0.5, loc=mu, scale=sigma)
        p_over[:, j] = np.clip(p_j, 0.0, 1.0)

    # Central 50%, 80%, 90% intervals from discretized Normal quantiles
    cov = {}
    widths = {}
    for conf_label, alpha in [("50%", 0.25), ("80%", 0.10), ("90%", 0.05)]:
        lo = np.maximum(0.0, np.rint(stats.norm.ppf(alpha, loc=mu, scale=sigma)))
        hi = np.maximum(0.0, np.rint(stats.norm.ppf(1.0 - alpha, loc=mu, scale=sigma)))
        cov[conf_label] = float(np.mean((actual >= lo) & (actual <= hi)))
        widths[conf_label] = float(np.mean(hi - lo))

    return {
        "p_over_mat": p_over,
        "coverage": cov,
        "interval_width": widths,
        "mu": mu,
        "sigma": sigma,
        "tier_counts": {"global_normal": n_rows},
        "support_mean": float(len(hist_res)),
    }


def evaluate_global_bootstrap_batch(
    pred: np.ndarray,
    actual: np.ndarray,
    lines_mat: np.ndarray,
    hist_res: np.ndarray,
    n_simulations: int = 10_000,
    seed: int = 42,
    chunk_size: int = 500,
) -> Dict[str, Any]:
    """Evaluate Method B (Global Residual Bootstrap, 10,000 draws/row) in memory-safe chunks."""
    rng = np.random.default_rng(seed)
    n_rows, n_lines = lines_mat.shape
    n_hist = len(hist_res)

    p_over = np.empty((n_rows, n_lines), dtype=float)
    cov_hits = {"50%": 0, "80%": 0, "90%": 0}
    width_sums = {"50%": 0.0, "80%": 0.0, "90%": 0.0}
    q_idx = [
        int(0.05 * (n_simulations - 1)),
        int(0.10 * (n_simulations - 1)),
        int(0.25 * (n_simulations - 1)),
        int(0.75 * (n_simulations - 1)),
        int(0.90 * (n_simulations - 1)),
        int(0.95 * (n_simulations - 1)),
    ]

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        m = end - start
        idx = rng.integers(0, n_hist, size=(m, n_simulations))
        sim = np.maximum(0, np.rint(pred[start:end, None] + hist_res[idx])).astype(np.int32)

        for j in range(n_lines):
            p_over[start:end, j] = np.mean(sim > lines_mat[start:end, j : j + 1], axis=1)

        sim_sorted = np.sort(sim, axis=1)
        act_c = actual[start:end]
        q05, q10, q25, q75, q90, q95 = (sim_sorted[:, pos] for pos in q_idx)
        cov_hits["90%"] += int(np.sum((act_c >= q05) & (act_c <= q95)))
        cov_hits["80%"] += int(np.sum((act_c >= q10) & (act_c <= q90)))
        cov_hits["50%"] += int(np.sum((act_c >= q25) & (act_c <= q75)))
        width_sums["90%"] += float(np.sum(q95 - q05))
        width_sums["80%"] += float(np.sum(q90 - q10))
        width_sums["50%"] += float(np.sum(q75 - q25))

    cov = {k: v / n_rows for k, v in cov_hits.items()}
    widths = {k: v / n_rows for k, v in width_sums.items()}
    return {
        "p_over_mat": p_over,
        "coverage": cov,
        "interval_width": widths,
        "tier_counts": {"global_bootstrap": n_rows},
        "support_mean": float(n_hist),
    }


def evaluate_conditional_bootstrap_batch(
    stat: str,
    df_query: pd.DataFrame,
    df_hist: pd.DataFrame,
    lines_mat: np.ndarray,
    config: ConditionalConfig,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate Method C (Conditional Residual Bootstrap, 10,000 draws/row) using
    O(log N) binary-search ResidualIndex slices and O(N) discrete integer bincount CDFs.
    """
    res_index = ResidualIndex.from_dataframe(df_hist, stat=stat, config=config)

    pred = df_query["predicted_value"].to_numpy(dtype=float)
    pred_min = df_query["predicted_minutes"].to_numpy(dtype=float)
    actual = df_query["actual_value"].to_numpy(dtype=float)

    if "ROLL5_STARTER_PCT" in df_query.columns and df_query["ROLL5_STARTER_PCT"].notna().any():
        starter_arr: Optional[np.ndarray] = (df_query["ROLL5_STARTER_PCT"].fillna(0.0) >= 0.5).to_numpy()
    elif "ROLL5_MIN_MEAN" in df_query.columns:
        starter_arr = (df_query["ROLL5_MIN_MEAN"].fillna(0.0) >= 24.0).to_numpy()
    else:
        starter_arr = None

    rng = np.random.default_rng(seed)
    n_rows, n_lines = lines_mat.shape
    n_sim = config.n_simulations
    p_over = np.empty((n_rows, n_lines), dtype=float)
    cov_hits = {"50%": 0, "80%": 0, "90%": 0}
    width_sums = {"50%": 0.0, "80%": 0.0, "90%": 0.0}
    support_sizes = np.empty(n_rows, dtype=int)
    tiers: List[str] = []

    q_targets = np.array([0.05, 0.10, 0.25, 0.75, 0.90, 0.95]) * n_sim
    k_lines_mat = np.floor(lines_mat).astype(int)
    has_starter = starter_arr is not None

    for i in range(n_rows):
        st_i = bool(starter_arr[i]) if has_starter else None
        cond_res, tier, _ = res_index.select_residuals(
            predicted_value=float(pred[i]),
            predicted_minutes=float(pred_min[i]),
            is_starter=st_i,
        )
        tiers.append(tier)
        n_cand = len(cond_res)
        support_sizes[i] = n_cand

        sampled_res = cond_res[rng.integers(0, n_cand, size=n_sim)]
        sim = np.maximum(0, np.rint(pred[i] + sampled_res)).astype(np.int32)

        # Discrete integer CDF via fast O(n_sim) bincount + cumsum (no sorting overhead)
        cum = np.cumsum(np.bincount(sim))
        max_k = len(cum) - 1

        for j in range(n_lines):
            k_le = int(k_lines_mat[i, j])
            if k_le < 0:
                p_over[i, j] = 1.0
            elif k_le >= max_k:
                p_over[i, j] = 0.0
            else:
                p_over[i, j] = 1.0 - (cum[k_le] / n_sim)

        qs = np.searchsorted(cum, q_targets, side="left")
        a_i = actual[i]
        if qs[0] <= a_i <= qs[5]:
            cov_hits["90%"] += 1
        if qs[1] <= a_i <= qs[4]:
            cov_hits["80%"] += 1
        if qs[2] <= a_i <= qs[3]:
            cov_hits["50%"] += 1
        width_sums["90%"] += float(qs[5] - qs[0])
        width_sums["80%"] += float(qs[4] - qs[1])
        width_sums["50%"] += float(qs[3] - qs[2])

    tier_counts = pd.Series(tiers).value_counts().to_dict()
    cov = {k: v / n_rows for k, v in cov_hits.items()}
    widths = {k: v / n_rows for k, v in width_sums.items()}

    return {
        "p_over_mat": p_over,
        "coverage": cov,
        "interval_width": widths,
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "tiers": tiers,
        "support_sizes": support_sizes,
        "support_mean": float(np.mean(support_sizes)),
        "support_min": int(np.min(support_sizes)),
    }


def summarize_prob_metrics(
    actual: np.ndarray,
    lines_mat: np.ndarray,
    p_over_mat: np.ndarray,
) -> Dict[str, Any]:
    """Compute Anchor Line (col 1) and Multi-Line Brier Score, Log Loss, ECE, Mean Pred vs Actual Hit Rate."""
    # Anchor line is column index 1 in compute_multi_lines_matrix; support 1-col too
    anchor_col = 1 if lines_mat.shape[1] >= 2 else 0
    y_anchor = (actual > lines_mat[:, anchor_col]).astype(int)
    p_anchor = p_over_mat[:, anchor_col]
    _, ece_anchor = compute_reliability_bins(y_anchor, p_anchor, n_bins=10)

    y_all = (actual[:, None] > lines_mat).ravel().astype(int)
    p_all = p_over_mat.ravel()
    bins_all, ece_all = compute_reliability_bins(y_all, p_all, n_bins=10)

    return {
        "anchor_brier": round(compute_brier_score(y_anchor, p_anchor), 6),
        "anchor_logloss": round(compute_log_loss(y_anchor, p_anchor), 6),
        "anchor_ece": round(ece_anchor, 5),
        "anchor_mean_pred_prob": round(float(np.mean(p_anchor)), 5),
        "anchor_actual_over_rate": round(float(np.mean(y_anchor)), 5),
        "multi_brier": round(compute_brier_score(y_all, p_all), 6),
        "multi_logloss": round(compute_log_loss(y_all, p_all), 6),
        "multi_ece": round(ece_all, 5),
        "multi_mean_pred_prob": round(float(np.mean(p_all)), 5),
        "multi_actual_over_rate": round(float(np.mean(y_all)), 5),
        "reliability_bins_multi": bins_all,
    }


# ── Phase 2 Diagnostic Plots ──────────────────────────────────────────────────

def create_phase2_plots(oos_22_25: Dict[str, pd.DataFrame], phase2_stats: Dict[str, Any]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.25})

    # 1. Residual Histograms with Normal Overlay
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.ravel()
    for i, stat in enumerate(TARGETS):
        ax = axes_flat[i]
        res = oos_22_25[stat]["residual"].to_numpy()
        s = phase2_stats[stat]
        q_lo, q_hi = np.quantile(res, [0.005, 0.995])
        bins = np.linspace(q_lo, q_hi, 45)
        ax.hist(res[(res >= q_lo) & (res <= q_hi)], bins=bins, density=True, alpha=0.65, color="#2563eb", label="OOS Residuals")
        xs = np.linspace(q_lo, q_hi, 250)
        pdf = stats.norm.pdf(xs, loc=s["mean_residual"], scale=s["std_residual"])
        ax.plot(xs, pdf, "r--", lw=1.8, label="Normal Fit")
        ax.axvline(0.0, color="black", lw=1.0, alpha=0.7)
        ax.axvline(s["mean_residual"], color="#dc2626", lw=1.1, ls=":")
        ax.set_title(f"{stat} (skew={s['skewness']:.2f}, kurt={s['excess_kurtosis']:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Residual (Actual - Pred)")
        if i == 0:
            ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    axes_flat[-1].text(
        0.05, 0.5,
        "Phase 2 OOS Residuals\n(2022-23 to 2024-25, n=72,387/stat)\n\n"
        "• Blue: Empirical residuals\n• Red dashed: Fitted Normal\n"
        "• Right skew & zero spikes show\n  why Normal fails on counts.",
        va="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    fig.suptitle("Residual Distributions vs. Normal Approximation (2022–25 Out-of-Sample)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_DIR / "residual_histograms_normal_overlay.png")
    plt.close(fig)

    # 2. Predicted vs Actual Scatter + Binned Empirical Mean
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.ravel()
    rng = np.random.default_rng(42)
    for i, stat in enumerate(TARGETS):
        ax = axes_flat[i]
        df = oos_22_25[stat]
        sub_idx = rng.choice(len(df), size=min(4000, len(df)), replace=False)
        p = df["predicted_value"].to_numpy()
        a = df["actual_value"].to_numpy()
        ax.scatter(p[sub_idx], a[sub_idx], s=6, alpha=0.18, color="#3b82f6")
        max_v = max(np.quantile(p, 0.995), np.quantile(a, 0.995))
        ax.plot([0, max_v], [0, max_v], "k--", lw=1.2, label="y = x (Ideal)")
        # Binned conditional expectation
        q_bins = np.unique(np.quantile(p, np.linspace(0, 1, 15)))
        if len(q_bins) > 3:
            b_idx = np.digitize(p, q_bins[1:-1])
            bx = [np.mean(p[b_idx == k]) for k in range(len(q_bins) - 1) if np.any(b_idx == k)]
            by = [np.mean(a[b_idx == k]) for k in range(len(q_bins) - 1) if np.any(b_idx == k)]
            ax.plot(bx, by, "r-o", ms=3, lw=1.6, label="Empirical Mean")
        ax.set_title(f"{stat}: Predicted vs Actual (mean bias={phase2_stats[stat]['mean_residual']:+.2f})", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        if i == 0:
            ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle("Predicted vs. Actual Outcomes & L1 Median-Bias Signature (2022–25 OOS)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_DIR / "predicted_vs_actual_scatter.png")
    plt.close(fig)

    # 3. Residuals vs Predicted Value (Heteroscedasticity Envelope)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.ravel()
    for i, stat in enumerate(TARGETS):
        ax = axes_flat[i]
        df = oos_22_25[stat]
        sub_idx = rng.choice(len(df), size=min(4000, len(df)), replace=False)
        p = df["predicted_value"].to_numpy()
        r = df["residual"].to_numpy()
        ax.scatter(p[sub_idx], r[sub_idx], s=6, alpha=0.15, color="#0d9488")
        ax.axhline(0.0, color="black", lw=1.1)
        q_bins = np.unique(np.quantile(p, np.linspace(0, 1, 12)))
        if len(q_bins) > 3:
            b_idx = np.digitize(p, q_bins[1:-1])
            bx, b_q10, b_q50, b_q90 = [], [], [], []
            for k in range(len(q_bins) - 1):
                m = b_idx == k
                if np.sum(m) >= 20:
                    bx.append(np.mean(p[m]))
                    b_q10.append(np.quantile(r[m], 0.10))
                    b_q50.append(np.quantile(r[m], 0.50))
                    b_q90.append(np.quantile(r[m], 0.90))
            ax.plot(bx, b_q50, "r-", lw=1.6, label="Median")
            ax.plot(bx, b_q90, "r--", lw=1.2, label="P10 / P90")
            ax.plot(bx, b_q10, "r--", lw=1.2)
        ax.set_title(f"{stat} (corr pred vs |res|={phase2_stats[stat]['corr_pred_vs_abs_residual']:.2f})", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual")
        if i == 0:
            ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle("Residual Spread vs. Predicted Stat Value (Heteroscedasticity Check)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_DIR / "residuals_vs_predicted.png")
    plt.close(fig)

    # 4. Residuals Grouped by Expected Minutes Bucket
    buckets = ["<15m", "15-25m", "25-32m", "32+m"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(buckets))
    w = 0.11
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(TARGETS)))
    for idx_s, stat in enumerate(TARGETS):
        b_map = phase2_stats[stat]["by_minutes_bucket"]
        sds = [b_map.get(b, {}).get("std_res", 0.0) for b in buckets]
        means = [b_map.get(b, {}).get("mean_res", 0.0) for b in buckets]
        offset = (idx_s - 3) * w
        axes[0].bar(x + offset, sds, width=w, label=stat, color=colors[idx_s])
        axes[1].plot(x, means, marker="o", label=stat, color=colors[idx_s], lw=1.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets)
    axes[0].set_ylabel("Residual Std Dev")
    axes[0].set_title("Residual Spread Increases Strongly with Expected Minutes", fontweight="bold")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buckets)
    axes[1].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[1].set_ylabel("Mean Residual Bias (Actual - Pred)")
    axes[1].set_title("Systematic Mean Residual by Expected Minutes Bucket", fontweight="bold")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Minutes Context Dependency of Prediction Errors (2022–25 OOS)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_DIR / "residuals_by_minutes_bucket.png")
    plt.close(fig)

    # 5. Error Distributions by Player Role (Starter vs Bench)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(TARGETS))
    starter_mae = [phase2_stats[s]["by_role"]["Starters"]["mae"] for s in TARGETS]
    bench_mae = [phase2_stats[s]["by_role"]["Bench"]["mae"] for s in TARGETS]
    starter_std = [phase2_stats[s]["by_role"]["Starters"]["std_res"] for s in TARGETS]
    bench_std = [phase2_stats[s]["by_role"]["Bench"]["std_res"] for s in TARGETS]
    starter_zero = [phase2_stats[s]["by_role"]["Starters"]["pct_zero_actual"] for s in TARGETS]
    bench_zero = [phase2_stats[s]["by_role"]["Bench"]["pct_zero_actual"] for s in TARGETS]

    for ax, st_vals, bn_vals, title, ylabel in [
        (axes[0], starter_mae, bench_mae, "MAE: Starters vs Bench", "Mean Absolute Error"),
        (axes[1], starter_std, bench_std, "Residual SD: Starters vs Bench", "Residual Std Dev"),
        (axes[2], starter_zero, bench_zero, "Zero-Outcome Rate (%)", "% Actual == 0"),
    ]:
        ax.bar(x - 0.18, st_vals, width=0.35, label="Starters", color="#1d4ed8")
        ax.bar(x + 0.18, bn_vals, width=0.35, label="Bench", color="#f59e0b")
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8.5)
    fig.suptitle("Error & Zero-Inflation Profile by Player Role (2022–25 OOS)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_DIR / "error_distribution_by_role.png")
    plt.close(fig)
    logger.info("Saved 5 Phase 2 diagnostic plots to %s", PLOT_DIR)


# ── Phase 3 & 6 Comparison / Reliability Plots ───────────────────────────────

def create_method_and_coverage_plots(
    method_comp: Dict[str, Any],
    winning_cfg_name: str,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TARGETS))
    w = 0.26

    # 1. Method Comparison Brier & Log Loss on 2024-25
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    methods = [
        ("normal", "Method A: Normal", "#ef4444"),
        ("global_bootstrap", "Method B: Global Bootstrap", "#f59e0b"),
        ("conditional_bootstrap", f"Method C: Conditional ({winning_cfg_name})", "#10b981"),
    ]
    for idx_m, (m_key, m_label, color) in enumerate(methods):
        briers = [method_comp[s]["methods"][m_key]["metrics"]["multi_brier"] for s in TARGETS]
        lls = [method_comp[s]["methods"][m_key]["metrics"]["multi_logloss"] for s in TARGETS]
        axes[0].bar(x + (idx_m - 1) * w, briers, width=w, label=m_label, color=color)
        axes[1].bar(x + (idx_m - 1) * w, lls, width=w, label=m_label, color=color)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TARGETS)
    axes[0].set_ylabel("Brier Score (Lower is Better)")
    axes[0].set_title("Multi-Line Brier Score by Method (2024–25 OOS)", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TARGETS)
    axes[1].set_ylabel("Log Loss (Lower is Better)")
    axes[1].set_title("Multi-Line Log Loss by Method (2024–25 OOS)", fontweight="bold")
    axes[1].legend(fontsize=8)
    fig.suptitle("Phase 3 Probability Method Showdown on Strictly Out-of-Sample 2024–25 Games", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_DIR / "method_comparison_brier_logloss.png")
    plt.close(fig)

    # 2. 50% / 80% / 90% Interval Coverage Comparison on 2024-25
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    for idx_c, (conf_lbl, nominal) in enumerate([("50%", 0.50), ("80%", 0.80), ("90%", 0.90)]):
        ax = axes[idx_c]
        for idx_m, (m_key, m_label, color) in enumerate(methods):
            vals = [method_comp[s]["methods"][m_key]["coverage"][conf_lbl] * 100.0 for s in TARGETS]
            ax.bar(x + (idx_m - 1) * w, vals, width=w, label=m_label, color=color)
        ax.axhline(nominal * 100.0, color="black", ls="--", lw=1.4, label=f"Nominal {conf_lbl}")
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS)
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_title(f"{conf_lbl} Prediction Interval Coverage", fontweight="bold")
        if idx_c == 0:
            ax.legend(fontsize=7.5)
    fig.suptitle("Empirical vs. Nominal Prediction Interval Coverage (2024–25 OOS)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_DIR / "interval_coverage_by_method.png")
    plt.close(fig)


def create_reliability_plot(backtest_results: Dict[str, Any], season_label: str, filename: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.ravel()
    for i, stat in enumerate(TARGETS):
        ax = axes_flat[i]
        res_s = backtest_results[stat]
        raw_bins = res_s["raw_metrics"]["reliability_bins_multi"]
        final_bins = res_s["final_metrics"]["reliability_bins_multi"]

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration")
        if raw_bins:
            ax.plot(
                [b["mean_predicted_prob"] for b in raw_bins],
                [b["actual_frequency"] for b in raw_bins],
                "o--", color="#64748b", lw=1.4, ms=4, label=f"Raw (ECE={res_s['raw_metrics']['multi_ece']:.3f})",
            )
        if final_bins:
            ax.plot(
                [b["mean_predicted_prob"] for b in final_bins],
                [b["actual_frequency"] for b in final_bins],
                "s-", color="#2563eb", lw=1.8, ms=4.5, label=f"Final (ECE={res_s['final_metrics']['multi_ece']:.3f})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{stat} ({res_s['calibration_method']})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Hit Rate")
        ax.legend(fontsize=7.5, loc="upper left")
    axes_flat[-1].axis("off")
    fig.suptitle(f"Calibration Reliability Diagrams — {season_label} Backtest", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_DIR / filename)
    plt.close(fig)


# ── Subgroup & Calibration Helpers ────────────────────────────────────────────

def compute_subgroup_backtest(
    df: pd.DataFrame,
    lines_mat: np.ndarray,
    p_final: np.ndarray,
) -> Dict[str, Any]:
    """Compute subgroup metrics by predicted minutes, starter role, pred quartile, and prob bin."""
    actual = df["actual_value"].to_numpy(dtype=float)
    anchor_line = lines_mat[:, 1]
    y_anchor = (actual > anchor_line).astype(int)
    p_anchor = p_final[:, 1]

    out: Dict[str, Any] = {}

    # 1. Minutes buckets
    min_bins = pd.cut(df["predicted_minutes"], bins=[0, 15, 25, 32, 60], labels=["<15m", "15-25m", "25-32m", "32+m"], right=False)
    min_res = {}
    for b in ["<15m", "15-25m", "25-32m", "32+m"]:
        m = (min_bins == b).to_numpy()
        if m.sum() >= 25 and len(np.unique(y_anchor[m])) >= 2:
            _, ece = compute_reliability_bins(y_anchor[m], p_anchor[m], n_bins=10)
            min_res[b] = {
                "n": int(m.sum()),
                "brier": round(compute_brier_score(y_anchor[m], p_anchor[m]), 5),
                "logloss": round(compute_log_loss(y_anchor[m], p_anchor[m]), 5),
                "ece": round(ece, 4),
                "mean_pred": round(float(np.mean(p_anchor[m])), 4),
                "actual_rate": round(float(np.mean(y_anchor[m])), 4),
            }
    out["by_minutes_bucket"] = min_res

    # 2. Role
    is_starter = (
        (df["ROLL5_STARTER_PCT"] >= 0.5)
        if "ROLL5_STARTER_PCT" in df.columns and df["ROLL5_STARTER_PCT"].notna().any()
        else (df["ROLL5_MIN_MEAN"] >= 24.0)
    ).to_numpy()
    role_res = {}
    for label, m in [("Starters", is_starter), ("Bench", ~is_starter)]:
        if m.sum() >= 25 and len(np.unique(y_anchor[m])) >= 2:
            _, ece = compute_reliability_bins(y_anchor[m], p_anchor[m], n_bins=10)
            role_res[label] = {
                "n": int(m.sum()),
                "brier": round(compute_brier_score(y_anchor[m], p_anchor[m]), 5),
                "logloss": round(compute_log_loss(y_anchor[m], p_anchor[m]), 5),
                "ece": round(ece, 4),
                "mean_pred": round(float(np.mean(p_anchor[m])), 4),
                "actual_rate": round(float(np.mean(y_anchor[m])), 4),
            }
    out["by_role"] = role_res

    # 3. Predicted Value Ranges (tertiles/quartiles)
    pred_vals = df["predicted_value"]
    try:
        pq = pd.qcut(pred_vals, q=4, duplicates="drop")
        pred_res = {}
        for cat in pq.cat.categories:
            m = (pq == cat).to_numpy()
            if m.sum() >= 25 and len(np.unique(y_anchor[m])) >= 2:
                _, ece = compute_reliability_bins(y_anchor[m], p_anchor[m], n_bins=8)
                pred_res[str(cat)] = {
                    "n": int(m.sum()),
                    "brier": round(compute_brier_score(y_anchor[m], p_anchor[m]), 5),
                    "ece": round(ece, 4),
                    "mean_pred": round(float(np.mean(p_anchor[m])), 4),
                    "actual_rate": round(float(np.mean(y_anchor[m])), 4),
                }
        out["by_predicted_range"] = pred_res
    except Exception:
        out["by_predicted_range"] = {}

    return out


def evaluate_fixed_milestones(
    df: pd.DataFrame,
    stat: str,
    res_index_hist: pd.DataFrame,
    method: str,
    cond_cfg: ConditionalConfig,
    calibrator: ProbabilityCalibrator,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate fixed domain milestone lines (e.g. PTS>19.5, 3PM>1.5, BLK>0.5)."""
    milestones = MILESTONE_LINES[stat]
    n = len(df)
    lines_mat = np.tile(np.array(milestones, dtype=float), (n, 1))
    actual = df["actual_value"].to_numpy(dtype=float)

    if method == "normal":
        out = evaluate_normal_batch(df["predicted_value"].to_numpy(), actual, lines_mat, res_index_hist["residual"].to_numpy(), seed=seed)
    elif method == "global_bootstrap":
        out = evaluate_global_bootstrap_batch(df["predicted_value"].to_numpy(), actual, lines_mat, res_index_hist["residual"].to_numpy(), n_simulations=cond_cfg.n_simulations, seed=seed)
    else:
        out = evaluate_conditional_bootstrap_batch(stat, df, res_index_hist, lines_mat, config=cond_cfg, seed=seed)

    p_raw = out["p_over_mat"]
    p_cal = np.column_stack([calibrator.transform(p_raw[:, j]) for j in range(len(milestones))])

    per_line = {}
    for j, line in enumerate(milestones):
        yj = (actual > line).astype(int)
        pj = p_cal[:, j]
        if len(np.unique(yj)) >= 2:
            _, ece_j = compute_reliability_bins(yj, pj, n_bins=10)
            per_line[f">{line}"] = {
                "brier": round(compute_brier_score(yj, pj), 5),
                "logloss": round(compute_log_loss(yj, pj), 5),
                "ece": round(ece_j, 4),
                "mean_pred": round(float(np.mean(pj)), 4),
                "actual_rate": round(float(np.mean(yj)), 4),
            }
    return per_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Version 1 NBA Probability Engine Validation Pipeline")
    parser.add_argument(
        "--phase6-only",
        action="store_true",
        help="Skip Phases 2-5 if artefacts/probability/engine_config.json & calibrators.joblib already exist.",
    )
    parser.add_argument(
        "--phase5-onward",
        action="store_true",
        help="Skip Phases 2-4 if engine_config.json exists, re-run Phase 5 calibration + Phase 6-7 cleanly.",
    )
    args = parser.parse_args()

    t0 = time.time()
    PROB_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── PHASE 1: Out-of-Sample Dataset Verification / Generation ──────────────
    logger.info("=== PHASE 1: Out-of-Sample Predictions & Residuals ===")
    oos_paths = generate_out_of_sample_predictions()
    oos_all: Dict[str, pd.DataFrame] = {}
    oos_22_25: Dict[str, pd.DataFrame] = {}
    phase1_summary: Dict[str, Any] = {}

    for stat in TARGETS:
        df = load_oos_predictions(stat)
        assert not df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).any(), f"Duplicate rows found in {stat}"
        assert df["predicted_value"].notna().all() and (df["predicted_value"] >= 0).all()
        assert df["residual"].notna().all()
        oos_all[stat] = df
        oos_22_25[stat] = df[df["SEASON"].isin(["2022-23", "2023-24", "2024-25"])].copy()
        phase1_summary[stat] = {
            "total_rows": int(len(df)),
            "by_season": {str(k): int(v) for k, v in df["SEASON"].value_counts().sort_index().items()},
            "model_versions": {str(k): int(v) for k, v in df["model_version"].value_counts().items()},
        }
        logger.info("Loaded %s: %d OOS rows across seasons %s", stat, len(df), list(phase1_summary[stat]["by_season"].keys()))

    cfg_file = PROB_ARTEFACT_DIR / "engine_config.json"
    cal_file = PROB_ARTEFACT_DIR / "calibrators.joblib"
    if args.phase6_only and cfg_file.exists() and cal_file.exists():
        logger.info("--phase6-only active: loading saved engine_config.json and calibrators.joblib ...")
        with open(cfg_file, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        selected_methods = loaded_cfg["selected_methods"]
        winning_cfg = ConditionalConfig(**loaded_cfg["conditional_config"])
        calibrators = joblib.load(cal_file)
    elif args.phase5_onward and cfg_file.exists():
        logger.info("--phase5-onward active: loading saved Phase 3/4 selection from engine_config.json & re-running Phases 5–7 ...")
        with open(cfg_file, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        best_cfg_name = loaded_cfg.get("selected_cfg_preset", "tight")
        selected_methods = loaded_cfg["selected_methods"]
        winning_cfg = ConditionalConfig(**loaded_cfg["conditional_config"])

        raw_sel_preds_24_25 = {}
        for stat in TARGETS:
            df = oos_all[stat]
            df_hist = df[df["SEASON"].isin(["2022-23", "2023-24"])].copy()
            df_sel = df[df["SEASON"] == "2024-25"].sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).copy()
            lines_mat = compute_multi_lines_matrix(df_sel, stat)
            actual = df_sel["actual_value"].to_numpy(dtype=float)
            pred = df_sel["predicted_value"].to_numpy(dtype=float)
            hist_res = df_hist["residual"].to_numpy(dtype=float)
            win_m = selected_methods[stat]
            if win_m == "normal":
                out_sel = evaluate_normal_batch(pred, actual, lines_mat, hist_res, seed=42)
            elif win_m == "global_bootstrap":
                out_sel = evaluate_global_bootstrap_batch(pred, actual, lines_mat, hist_res, n_simulations=winning_cfg.n_simulations, seed=42)
            else:
                out_sel = evaluate_conditional_bootstrap_batch(stat, df_sel, df_hist, lines_mat, config=winning_cfg, seed=42)
            y_mat = (actual[:, None] > lines_mat).astype(int)
            raw_sel_preds_24_25[stat] = (out_sel["p_over_mat"].ravel(), y_mat.ravel())

        logger.info("=== PHASE 5: Chronological Calibration Fitting & Selection on 2024-25 ===")
        calibrators = {}
        phase5_report = {}
        for stat in TARGETS:
            p_raw_flat, y_flat = raw_sel_preds_24_25[stat]
            cal = ProbabilityCalibrator(stat=stat).fit_and_select(p_raw_flat, y_flat)
            calibrators[stat] = cal
            phase5_report[stat] = cal.to_metadata()
            logger.info(
                "%-3s | cal=%-8s (%s) | val_brier: raw=%.5f -> cal=%.5f | val_ll: raw=%.5f -> cal=%.5f",
                stat, cal.method, cal.status,
                cal.raw_val_brier or 0.0, cal.cal_val_brier or 0.0,
                cal.raw_val_logloss or 0.0, cal.cal_val_logloss or 0.0,
            )

        joblib.dump(calibrators, PROB_ARTEFACT_DIR / "calibrators.joblib")
        loaded_cfg["calibration_summary"] = phase5_report
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(loaded_cfg, f, indent=2)
        with open(PROB_ARTEFACT_DIR / "phase5_calibration_report.json", "w", encoding="utf-8") as f:
            json.dump(phase5_report, f, indent=2)
    else:
        # ── PHASE 2: Residual Analysis (Strictly 2022-25 OOS) ─────────────────────
        logger.info("=== PHASE 2: Residual Distribution Analysis (2022–25 OOS) ===")
        phase2_report: Dict[str, Any] = {}
        for stat in TARGETS:
            stats_dict = analyse_residual_distribution(oos_22_25[stat], stat)
            phase2_report[stat] = stats_dict
            logger.info(
                "%-3s | mean_res=%+.3f med=%+.3f std=%.3f mae=%.3f zero=%.1f%% skew=%.2f kurt=%.2f | %s",
                stat, stats_dict["mean_residual"], stats_dict["median_residual"], stats_dict["std_residual"],
                stats_dict["mae"], stats_dict["pct_zero_actual"], stats_dict["skewness"],
                stats_dict["excess_kurtosis"], stats_dict["shape_diagnosis"],
            )

        create_phase2_plots(oos_22_25, phase2_report)
        with open(PROB_ARTEFACT_DIR / "phase2_residual_analysis.json", "w", encoding="utf-8") as f:
            json.dump(phase2_report, f, indent=2)

        # ── PHASE 3 & 4: Config Validation & Method Comparison on 2024-25 OOS ─────
        logger.info("=== PHASE 3/4: Tuning ConditionalConfig & Comparing Methods A/B/C on 2024-25 ===")
        cfg_sweep_scores: Dict[str, Any] = {}
        cfg_sweep_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for cfg_name, cfg in CONFIG_CANDIDATES.items():
            brier_list, ll_list, ece_list, t3_list = [], [], [], []
            per_stat_cfg = {}
            cfg_sweep_cache[cfg_name] = {}
            for stat in TARGETS:
                df = oos_all[stat]
                df_hist = df[df["SEASON"].isin(["2022-23", "2023-24"])].copy()
                df_sel = df[df["SEASON"] == "2024-25"].sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).copy()
                lines_mat = compute_multi_lines_matrix(df_sel, stat)
                out_c = evaluate_conditional_bootstrap_batch(stat, df_sel, df_hist, lines_mat, config=cfg, seed=42)
                cfg_sweep_cache[cfg_name][stat] = out_c
                met = summarize_prob_metrics(df_sel["actual_value"].to_numpy(), lines_mat, out_c["p_over_mat"])
                t3_rate = out_c["tier_counts"].get("tier3_global_fallback", 0) / len(df_sel)
                brier_list.append(met["multi_brier"])
                ll_list.append(met["multi_logloss"])
                ece_list.append(met["multi_ece"])
                t3_list.append(t3_rate)
                per_stat_cfg[stat] = {
                    "multi_brier": met["multi_brier"],
                    "multi_logloss": met["multi_logloss"],
                    "multi_ece": met["multi_ece"],
                    "tier3_rate": round(t3_rate, 4),
                    "support_mean": round(out_c["support_mean"], 1),
                }
            cfg_sweep_scores[cfg_name] = {
                "mean_multi_brier": round(float(np.mean(brier_list)), 6),
                "mean_multi_logloss": round(float(np.mean(ll_list)), 6),
                "mean_multi_ece": round(float(np.mean(ece_list)), 5),
                "mean_tier3_rate": round(float(np.mean(t3_list)), 5),
                "per_stat": per_stat_cfg,
            }
            logger.info(
                "Config %-8s | mean_brier=%.6f mean_logloss=%.6f mean_ece=%.5f mean_t3=%.4f",
                cfg_name, cfg_sweep_scores[cfg_name]["mean_multi_brier"],
                cfg_sweep_scores[cfg_name]["mean_multi_logloss"],
                cfg_sweep_scores[cfg_name]["mean_multi_ece"],
                cfg_sweep_scores[cfg_name]["mean_tier3_rate"],
            )

        best_cfg_name = min(
            cfg_sweep_scores.keys(),
            key=lambda k: (cfg_sweep_scores[k]["mean_multi_logloss"], cfg_sweep_scores[k]["mean_multi_brier"]),
        )
        winning_cfg = CONFIG_CANDIDATES[best_cfg_name]
        logger.info("Selected ConditionalConfig preset on 2024-25: %s", best_cfg_name)

        # Now full Method A vs B vs C comparison on 2024-25
        method_comparison: Dict[str, Any] = {}
        selected_methods = {}
        raw_sel_preds_24_25: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for stat in TARGETS:
            df = oos_all[stat]
            df_hist = df[df["SEASON"].isin(["2022-23", "2023-24"])].copy()
            df_sel = df[df["SEASON"] == "2024-25"].sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).copy()
            lines_mat = compute_multi_lines_matrix(df_sel, stat)
            actual = df_sel["actual_value"].to_numpy(dtype=float)
            pred = df_sel["predicted_value"].to_numpy(dtype=float)
            hist_res = df_hist["residual"].to_numpy(dtype=float)

            out_a = evaluate_normal_batch(pred, actual, lines_mat, hist_res, seed=42)
            out_b = evaluate_global_bootstrap_batch(pred, actual, lines_mat, hist_res, n_simulations=winning_cfg.n_simulations, seed=42)
            out_c = cfg_sweep_cache[best_cfg_name][stat]

            met_a = summarize_prob_metrics(actual, lines_mat, out_a["p_over_mat"])
            met_b = summarize_prob_metrics(actual, lines_mat, out_b["p_over_mat"])
            met_c = summarize_prob_metrics(actual, lines_mat, out_c["p_over_mat"])

            # Select winning method strictly on 2024-25 OOS multi-line Log Loss + Brier + ECE balance
            candidates_m = [
                ("normal", met_a["multi_logloss"], met_a["multi_brier"], out_a["p_over_mat"]),
                ("global_bootstrap", met_b["multi_logloss"], met_b["multi_brier"], out_b["p_over_mat"]),
                ("conditional_bootstrap", met_c["multi_logloss"], met_c["multi_brier"], out_c["p_over_mat"]),
            ]
            win_m, _, _, win_p_mat = min(candidates_m, key=lambda x: (x[1], x[2]))
            selected_methods[stat] = win_m

            # Store chronological row-interleaved (p_raw, y_true) for Phase 5 calibration
            y_mat = (actual[:, None] > lines_mat).astype(int)
            raw_sel_preds_24_25[stat] = (win_p_mat.ravel(), y_mat.ravel())

            method_comparison[stat] = {
                "winning_method": win_m,
                "methods": {
                    "normal": {
                        "metrics": met_a,
                        "coverage": out_a["coverage"],
                        "interval_width": out_a["interval_width"],
                    },
                    "global_bootstrap": {
                        "metrics": met_b,
                        "coverage": out_b["coverage"],
                        "interval_width": out_b["interval_width"],
                    },
                    "conditional_bootstrap": {
                        "metrics": met_c,
                        "coverage": out_c["coverage"],
                        "interval_width": out_c["interval_width"],
                        "tier_counts": out_c["tier_counts"],
                        "support_mean": round(out_c["support_mean"], 1),
                        "support_min": out_c["support_min"],
                    },
                },
            }
            logger.info(
                "%-3s | Winner=%-21s | LogLoss: A=%.4f B=%.4f C=%.4f | Brier: A=%.4f B=%.4f C=%.4f | ECE: A=%.4f B=%.4f C=%.4f",
                stat, win_m,
                met_a["multi_logloss"], met_b["multi_logloss"], met_c["multi_logloss"],
                met_a["multi_brier"], met_b["multi_brier"], met_c["multi_brier"],
                met_a["multi_ece"], met_b["multi_ece"], met_c["multi_ece"],
            )

        create_method_and_coverage_plots(method_comparison, best_cfg_name)
        with open(PROB_ARTEFACT_DIR / "phase3_method_comparison_2024_25.json", "w", encoding="utf-8") as f:
            json.dump({"config_sweep": cfg_sweep_scores, "best_config": best_cfg_name, "by_stat": method_comparison}, f, indent=2)

        # ── PHASE 5: Chronological Calibration on 2024-25 ─────────────────────────
        logger.info("=== PHASE 5: Chronological Calibration Fitting & Selection on 2024-25 ===")
        calibrators = {}
        phase5_report: Dict[str, Any] = {}
        for stat in TARGETS:
            p_raw_flat, y_flat = raw_sel_preds_24_25[stat]
            cal = ProbabilityCalibrator(stat=stat).fit_and_select(p_raw_flat, y_flat)
            calibrators[stat] = cal
            phase5_report[stat] = cal.to_metadata()
            logger.info(
                "%-3s | cal=%-8s (%s) | val_brier: raw=%.5f -> cal=%.5f | val_ll: raw=%.5f -> cal=%.5f",
                stat, cal.method, cal.status,
                cal.raw_val_brier or 0.0, cal.cal_val_brier or 0.0,
                cal.raw_val_logloss or 0.0, cal.cal_val_logloss or 0.0,
            )

        joblib.dump(calibrators, PROB_ARTEFACT_DIR / "calibrators.joblib")
        engine_config = {
            "version": "v1",
            "selected_cfg_preset": best_cfg_name,
            "conditional_config": {
                "pred_bandwidth_scale": winning_cfg.pred_bandwidth_scale,
                "min_bandwidth_scale": winning_cfg.min_bandwidth_scale,
                "tier2_pred_bandwidth_scale": winning_cfg.tier2_pred_bandwidth_scale,
                "tier2_min_bandwidth_scale": winning_cfg.tier2_min_bandwidth_scale,
                "min_samples_tier1": winning_cfg.min_samples_tier1,
                "min_samples_tier2": winning_cfg.min_samples_tier2,
                "max_knn_z_dist": winning_cfg.max_knn_z_dist,
                "match_starter_role": winning_cfg.match_starter_role,
                "n_simulations": winning_cfg.n_simulations,
                "random_seed": winning_cfg.random_seed,
            },
            "selected_methods": selected_methods,
            "calibration_summary": phase5_report,
        }
        with open(PROB_ARTEFACT_DIR / "engine_config.json", "w", encoding="utf-8") as f:
            json.dump(engine_config, f, indent=2)
        with open(PROB_ARTEFACT_DIR / "phase5_calibration_report.json", "w", encoding="utf-8") as f:
            json.dump(phase5_report, f, indent=2)

    # ── PHASE 6: Backtesting on 2025-26 & Unseen 2026-27 Holdout ──────────────
    def run_season_backtest(target_season: str, prior_seasons: List[str], out_json_name: str, plot_name: str) -> Dict[str, Any]:
        logger.info("=== PHASE 6: Backtesting on %s (residual library %s) ===", target_season, prior_seasons)
        results: Dict[str, Any] = {}
        for stat in TARGETS:
            df = oos_all[stat]
            df_hist = df[df["SEASON"].isin(prior_seasons)].copy()
            df_test = df[df["SEASON"] == target_season].sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).copy()
            lines_mat = compute_multi_lines_matrix(df_test, stat)
            actual = df_test["actual_value"].to_numpy(dtype=float)
            pred = df_test["predicted_value"].to_numpy(dtype=float)
            hist_res = df_hist["residual"].to_numpy(dtype=float)
            win_m = selected_methods[stat]

            if win_m == "normal":
                out = evaluate_normal_batch(pred, actual, lines_mat, hist_res, seed=42)
            elif win_m == "global_bootstrap":
                out = evaluate_global_bootstrap_batch(pred, actual, lines_mat, hist_res, n_simulations=winning_cfg.n_simulations, seed=42)
            else:
                out = evaluate_conditional_bootstrap_batch(stat, df_test, df_hist, lines_mat, config=winning_cfg, seed=42)

            p_raw = out["p_over_mat"]
            cal = calibrators[stat]
            p_final = np.column_stack([cal.transform(p_raw[:, j]) for j in range(p_raw.shape[1])])

            raw_met = summarize_prob_metrics(actual, lines_mat, p_raw)
            final_met = summarize_prob_metrics(actual, lines_mat, p_final)
            subgroups = compute_subgroup_backtest(df_test, lines_mat, p_final)
            milestones = evaluate_fixed_milestones(df_test, stat, df_hist, win_m, winning_cfg, cal, seed=42)

    # Check Integer Line Push Evaluation
            int_line = np.maximum(
                1.0,
                np.round(
                    df_test["ROLL10_STAT_MEAN"]
                    .fillna(pd.Series(pred, index=df_test.index))
                    .to_numpy(dtype=float)
                ),
            )
            actual_push_rate = float(np.mean(actual == int_line))

            results[stat] = {
                "season": target_season,
                "n_rows": int(len(df_test)),
                "method_used": win_m,
                "calibration_method": cal.method,
                "raw_metrics": raw_met,
                "final_metrics": final_met,
                "coverage": {k: round(v, 4) for k, v in out["coverage"].items()},
                "interval_width": {k: round(v, 3) for k, v in out["interval_width"].items()},
                "tier_counts": out["tier_counts"],
                "support_mean": round(out["support_mean"], 1),
                "integer_line_actual_push_rate": round(actual_push_rate, 4),
                "subgroups": subgroups,
                "milestones": milestones,
            }
            logger.info(
                "%-3s (%s, n=%d) | Brier=%.5f LogLoss=%.5f ECE=%.4f | Cov50=%.1f%% Cov80=%.1f%% Cov90=%.1f%%",
                stat, target_season, len(df_test),
                final_met["multi_brier"], final_met["multi_logloss"], final_met["multi_ece"],
                out["coverage"]["50%"] * 100.0, out["coverage"]["80%"] * 100.0, out["coverage"]["90%"] * 100.0,
            )

        with open(PROB_ARTEFACT_DIR / out_json_name, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        create_reliability_plot(results, f"{target_season} (n={results['PTS']['n_rows']})", plot_name)
        return results

    run_season_backtest(
        target_season="2025-26",
        prior_seasons=["2022-23", "2023-24", "2024-25"],
        out_json_name="phase6_backtest_2025_26.json",
        plot_name="reliability_diagrams_val_2025_26.png",
    )
    run_season_backtest(
        target_season="2026-27",
        prior_seasons=["2022-23", "2023-24", "2024-25", "2025-26"],
        out_json_name="phase6_backtest_2026_27.json",
        plot_name="reliability_diagrams_holdout_2026_27.png",
    )

    # ── PHASE 7: End-to-End Inference Verification ────────────────────────────
    logger.info("=== PHASE 7: PlayerProbabilityEngine End-to-End Query Verification ===")
    engine = PlayerProbabilityEngine()
    queries = [
        {
            "player": "Stephen Curry",
            "opponent": "Los Angeles Lakers",
            "is_home": 1,
            "lines_by_stat": {"PTS": [20.5, 25.0, 25.5, 30.5], "3PM": [3.5, 4.5], "AST": [4.5]},
        },
        {
            "player": "Nikola Jokic",
            "opponent": "Boston Celtics",
            "is_home": 1,
            "lines_by_stat": {"PTS": [24.5, 29.5], "REB": [10.5, 12.5], "AST": [8.5, 9.5]},
        },
        {
            "player": "LeBron James",
            "opponent": "Golden State Warriors",
            "is_home": 0,
            "lines_by_stat": {"PTS": [20.5, 25.5], "REB": [6.5], "AST": [6.5, 7.5], "BLK": [0.5], "STL": [0.5]},
        },
    ]

    live_examples = []
    for q in queries:
        try:
            res = engine.predict_player(
                player=q["player"],
                opponent=q["opponent"],
                is_home=q["is_home"],
                lines_by_stat=q["lines_by_stat"],
            )
            dist_summary = {}
            for s, dist in res["distributions"].items():
                lo90, hi90 = dist.interval(0.90)
                dist_summary[s] = {
                    "point_pred": res["predicted_stats"][s],
                    "sim_mean": round(dist.distribution_mean, 2),
                    "sim_median": round(dist.distribution_median, 2),
                    "sim_std": round(dist.distribution_std, 2),
                    "interval_90": [round(lo90, 1), round(hi90, 1)],
                    "tier": dist.fallback_tier,
                    "n_support": dist.n_support,
                }
            live_examples.append({
                "query": q,
                "metadata": res["metadata"],
                "predicted_minutes": res["predicted_minutes"],
                "is_starter_context": res["is_starter_context"],
                "predicted_stats": res["predicted_stats"],
                "distribution_summary": dist_summary,
                "line_probabilities": res["line_probabilities"],
            })
            logger.info(
                "Query %s vs %s | pred_min=%.1f | PTS=%.1f (line 25.5 over=%.1f%%)",
                res["metadata"].get("player_name"), res["metadata"].get("opponent_abbrev"),
                res["predicted_minutes"], res["predicted_stats"]["PTS"],
                (res["line_probabilities"].get("PTS", [{}])[1].get("prob_over", 0.0) if len(res["line_probabilities"].get("PTS", [])) > 1 else 0.0) * 100.0,
            )
        except Exception as e:
            logger.exception("Live example query failed for %s: %s", q["player"], e)

    with open(PROB_ARTEFACT_DIR / "phase7_live_examples.json", "w", encoding="utf-8") as f:
        json.dump(live_examples, f, indent=2)

    elapsed = time.time() - t0
    logger.info("=== PIPELINE COMPLETE in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
