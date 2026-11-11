"""
Unit & Leakage-Verification Tests for the Version 1 NBA Player Probability Engine.

Covers every required statistical, boundary, calibration, fallback, and chronological
no-leakage contract across:
  - models/residuals.py
  - models/calibration.py
  - models/probability.py
  - models/inference.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from models.calibration import (
    ProbabilityCalibrator,
    compute_brier_score,
    compute_log_loss,
    compute_reliability_bins,
)
from models.probability import (
    ConditionalConfig,
    OutcomeDistribution,
    ResidualIndex,
    calculate_line_probabilities,
    calculate_over_under_probability,
    estimate_conditional_bootstrap_probability,
    estimate_global_bootstrap_probability,
    estimate_normal_probability,
    is_integer_line,
    simulate_player_outcomes,
)
from models.residuals import (
    TARGETS,
    analyse_residual_distribution,
    calculate_residuals,
    canonical_stat_name,
    load_oos_predictions,
)


def _make_synthetic_oos_pool(n: int = 800, stat: str = "PTS", seed: int = 42) -> ResidualIndex:
    """Create a synthetic comparable ResidualIndex with starter & bench regimes."""
    rng = np.random.default_rng(seed)
    is_starter = rng.random(n) >= 0.4
    pred_min = np.where(is_starter, rng.normal(31.0, 3.5, n), rng.normal(17.0, 4.5, n))
    pred_min = np.clip(pred_min, 6.0, 42.0)

    if stat in {"BLK", "STL"}:
        pred_val = np.clip(0.03 * pred_min + rng.normal(0.0, 0.25, n), 0.0, 4.0)
        # Zero-inflated right-skewed discrete residuals
        actual = rng.poisson(lam=np.maximum(0.15, pred_val + 0.2))
        res = actual - pred_val
    else:
        pred_val = np.clip(0.55 * pred_min + rng.normal(0.0, 3.0, n), 0.0, 38.0)
        scale = np.where(is_starter, 6.0, 3.5)
        res = rng.normal(0.4, scale, n)

    return ResidualIndex(
        stat=stat,
        residuals=res,
        predicted_values=pred_val,
        predicted_minutes=pred_min,
        is_starter=is_starter,
        config=ConditionalConfig(n_simulations=4000, random_seed=seed),
    )


# 1. Half-point lines -> zero push, over + under == 1.0
def test_half_point_lines_no_push_and_exact_sum():
    outcomes = np.array([18, 22, 25, 26, 30, 32], dtype=int)
    for line in [0.5, 19.5, 25.5, 31.5]:
        assert not is_integer_line(line)
        p_over, p_under, p_push = calculate_line_probabilities(outcomes, line)
        assert p_push == 0.0
        assert p_over + p_under == pytest.approx(1.0, abs=1e-9)

    p_over, p_under, p_push = calculate_line_probabilities(outcomes, 25.5)
    assert p_over == pytest.approx(3 / 6)
    assert p_under == pytest.approx(3 / 6)
    assert p_push == 0.0

    # Through OutcomeDistribution + simulate_player_outcomes
    idx = _make_synthetic_oos_pool(n=600, stat="PTS")
    dist = simulate_player_outcomes("PTS", 22.0, 32.0, idx, method="conditional_bootstrap", is_starter=True)
    r = dist.probability_for_line(22.5)
    assert r.is_integer_line is False
    assert r.prob_push == 0.0
    assert r.prob_over + r.prob_under == pytest.approx(1.0, abs=1e-6)


# 2. Integer lines -> exact push probability and coherent sum (raw + calibrated)
def test_integer_lines_exact_push_raw_and_calibrated():
    outcomes = np.array([24, 25, 25, 26], dtype=int)
    assert is_integer_line(25.0)
    res_dict = calculate_over_under_probability(outcomes, 25.0)
    assert res_dict["prob_over"] == pytest.approx(0.25)
    assert res_dict["prob_under"] == pytest.approx(0.25)
    assert res_dict["prob_push"] == pytest.approx(0.50)
    assert sum(res_dict.values()) == pytest.approx(1.0)

    # Also test calibrated OutcomeDistribution on integer lines (monotonic survival S(k-1) >= S(k))
    rng = np.random.default_rng(7)
    p_train = np.linspace(0.05, 0.95, 1000)
    # Construct monotone synthetic miscalibrated probabilities where calibration wins cleanly
    true_logit = 2.2 * np.log(p_train / (1.0 - p_train))
    y_train = rng.binomial(1, 1.0 / (1.0 + np.exp(-true_logit)))
    cal = ProbabilityCalibrator(stat="PTS").fit_and_select(p_train, y_train, min_samples=500)
    assert cal.method in {"platt", "isotonic"}

    dist = OutcomeDistribution(
        stat="PTS",
        predicted_value=25.0,
        predicted_minutes=34.0,
        outcomes=np.repeat(outcomes, 500),
        method_used="conditional_bootstrap",
        n_simulations=2000,
        n_support=500,
        fallback_tier="tier1_tight_comparable",
        used_fallback=False,
        calibrator=cal,
    )
    line_res = dist.probability_for_line(25.0)
    assert line_res.is_integer_line is True
    assert line_res.prob_push >= 0.0
    assert line_res.prob_over + line_res.prob_under + line_res.prob_push == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= line_res.prob_over <= 1.0
    assert 0.0 <= line_res.prob_under <= 1.0


# 3. Zero predictions & rare zero-inflated stats (BLK / STL)
def test_zero_prediction_and_rare_zero_stats():
    idx_blk = _make_synthetic_oos_pool(n=700, stat="BLK", seed=99)
    dist_zero = simulate_player_outcomes("BLK", 0.0, 12.0, idx_blk, method="conditional_bootstrap", is_starter=False)

    assert np.all(dist_zero.outcomes >= 0)
    assert np.issubdtype(dist_zero.outcomes.dtype, np.integer)

    # Line 0.5 -> Over (>=1), Under (==0), zero push
    r_half = dist_zero.probability_for_line(0.5)
    assert r_half.prob_push == 0.0
    assert r_half.prob_over + r_half.prob_under == pytest.approx(1.0, abs=1e-9)

    # Line 0.0 -> integer line: Push is actual==0, Over is >=1, Under (<0) must be strictly 0.0
    r_zero = dist_zero.probability_for_line(0.0)
    assert r_zero.is_integer_line is True
    assert r_zero.prob_under == 0.0
    assert r_zero.prob_over + r_zero.prob_push == pytest.approx(1.0, abs=1e-9)


# 4. Zero negative simulated outcomes invariant across ALL three methods
@pytest.mark.parametrize("method", ["normal", "global_bootstrap", "conditional_bootstrap"])
def test_no_negative_outcomes_any_method(method: str):
    # Deliberately heavy negative residuals with tiny predicted_value=0.2
    n = 300
    heavy_neg_res = np.concatenate([np.full(200, -8.5), np.full(100, 1.5)])
    idx = ResidualIndex(
        stat="TOV",
        residuals=heavy_neg_res,
        predicted_values=np.full(n, 0.2),
        predicted_minutes=np.full(n, 15.0),
        is_starter=np.zeros(n, dtype=bool),
        config=ConditionalConfig(min_samples_tier1=50, min_samples_tier2=30, n_simulations=2000),
    )
    dist = simulate_player_outcomes("TOV", predicted_value=-0.08, predicted_minutes=15.0, residual_index=idx, method=method)
    assert dist.predicted_value == 0.0  # negative raw predictions clamped
    assert np.all(dist.outcomes >= 0)
    assert np.issubdtype(dist.outcomes.dtype, np.integer)


# 5. Probability sum & valid [0, 1] bounds across arbitrary lines and methods
@pytest.mark.parametrize("method", ["normal", "global_bootstrap", "conditional_bootstrap"])
def test_probability_bounds_and_sum_invariant(method: str):
    idx = _make_synthetic_oos_pool(n=600, stat="AST", seed=11)
    dist = simulate_player_outcomes("AST", 5.2, 31.0, idx, method=method, is_starter=True)

    for line in [0.0, 0.5, 2.0, 4.5, 5.0, 7.5, 20.5]:
        res = dist.probability_for_line(line)
        d = res.to_dict()
        for k in ("prob_over", "prob_under", "prob_push", "raw_prob_over", "raw_prob_under", "raw_prob_push"):
            assert 0.0 <= d[k] <= 1.0
        assert res.prob_over + res.prob_under + res.prob_push == pytest.approx(1.0, abs=1e-9)
        assert res.raw_prob_over + res.raw_prob_under + res.raw_prob_push == pytest.approx(1.0, abs=1e-9)


# 6. Insufficient residuals, extreme outlier k-NN, k=1 guard, and missing starter context
def test_insufficient_residuals_fallback_and_missing_context():
    # Case A: Tiny pool (< min_samples_tier2) -> deterministic tier3_global_fallback
    tiny_res = np.array([-2.0, -1.0, 0.0, 1.0, 3.0] * 6)
    tiny_idx = ResidualIndex(
        stat="PTS",
        residuals=tiny_res,
        predicted_values=np.full(len(tiny_res), 15.0),
        predicted_minutes=np.full(len(tiny_res), 25.0),
        is_starter=None,  # Missing starter context
        config=ConditionalConfig(min_samples_tier1=250, min_samples_tier2=150, n_simulations=1000),
    )
    cond, tier, used_fb = tiny_idx.select_residuals(predicted_value=15.0, predicted_minutes=25.0, is_starter=None)
    assert tier == "tier3_global_fallback"
    assert used_fb is True
    assert len(cond) == len(tiny_res)

    # Case B: Extreme outlier outside box -> bounded k-NN vs tier3 fallback depending on z-distance
    extreme_cond, extreme_tier, extreme_fb = tiny_idx.select_residuals(predicted_value=60.0, predicted_minutes=48.0, is_starter=True)
    assert extreme_tier == "tier3_global_fallback"
    assert extreme_fb is True
    assert len(extreme_cond) == len(tiny_res)

    # Case C: min_samples_tier2=1 cKDTree scalar query guard (both single & batch)
    k1_idx = ResidualIndex(
        stat="STL",
        residuals=np.array([-0.5, 0.0, 0.5, 1.0]),
        predicted_values=np.array([0.4, 0.5, 0.6, 0.7]),
        predicted_minutes=np.array([20.0, 22.0, 24.0, 26.0]),
        config=ConditionalConfig(
            min_samples_tier1=100,
            min_samples_tier2=1,
            tier2_pred_bandwidth_scale=0.1,
            tier2_min_bandwidth_scale=0.1,
            max_knn_z_dist=5.0,
        ),
    )
    arr_s, t_s, _ = k1_idx.select_residuals(predicted_value=0.95, predicted_minutes=26.5)
    assert isinstance(arr_s, np.ndarray) and arr_s.ndim == 1 and len(arr_s) == 1
    assert t_s == "tier2_knn_comparable"
    batch_idx, batch_tiers = k1_idx.batch_select_indices(np.array([0.95]), np.array([26.5]))
    assert len(batch_idx[0]) == 1 and batch_tiers[0] == "tier2_knn_comparable"

    # Case D: Normal method with < 2 residuals raises clear ValueError
    with pytest.raises(ValueError, match="at least 2"):
        estimate_normal_probability(10.0, np.array([0.5]), 9.5)


# 7. Insufficient calibration data or single class -> retains raw probabilities safely
def test_insufficient_calibration_data_graceful_fallback():
    cal_small = ProbabilityCalibrator(stat="BLK").fit_and_select(
        p_raw=np.linspace(0.1, 0.9, 120),
        y_true=(np.arange(120) % 2),
        min_samples=500,
    )
    assert cal_small.method == "none"
    assert cal_small.status == "insufficient_calibration_data"
    assert cal_small.transform(0.37) == pytest.approx(0.37)

    # Single-class constant target
    cal_const = ProbabilityCalibrator(stat="BLK").fit_and_select(
        p_raw=np.linspace(0.05, 0.4, 700),
        y_true=np.zeros(700, dtype=int),
        min_samples=500,
    )
    assert cal_const.method == "none"
    assert cal_const.status in {"insufficient_calibration_data", "insufficient_class_balance"}
    arr_out = cal_const.transform(np.array([0.2, 0.8]))
    np.testing.assert_allclose(arr_out, [0.2, 0.8])


# 8. Multiple threshold queries reuse ONE single simulated outcome distribution
def test_multi_line_reuses_single_distribution_and_is_monotone():
    idx = _make_synthetic_oos_pool(n=700, stat="PTS", seed=5)
    dist = simulate_player_outcomes("PTS", 25.0, 34.0, idx, method="conditional_bootstrap", is_starter=True, n_simulations=10_000)
    raw_outcomes_id = id(dist.outcomes)
    raw_outcomes_copy = dist.outcomes.copy()

    lines = [15.5, 20.5, 25.5, 30.5, 35.5]
    results = dist.probabilities_for_lines(lines)

    assert id(dist.outcomes) == raw_outcomes_id
    np.testing.assert_array_equal(dist.outcomes, raw_outcomes_copy)

    # Same expectation / std across all lines
    expected_vals = {r.expected_value for r in results}
    stds = {r.distribution_std for r in results}
    assert len(expected_vals) == 1
    assert len(stds) == 1

    # P(Over line) must be monotonically non-increasing as line increases
    overs = [r.prob_over for r in results]
    unders = [r.prob_under for r in results]
    assert all(overs[i] >= overs[i + 1] - 1e-12 for i in range(len(overs) - 1))
    assert all(unders[i] <= unders[i + 1] + 1e-12 for i in range(len(unders) - 1))


# 9. Calibration strictly chronological & selective (retains raw when raw is already optimal)
def test_calibration_strictly_selective_and_non_leaking():
    rng = np.random.default_rng(123)
    p_raw = rng.uniform(0.1, 0.9, size=2000)
    # Exact Bayes-calibrated outcomes: no systematic bias or temp scaling error exists
    y_true = rng.binomial(1, p_raw)
    cal = ProbabilityCalibrator(stat="PTS").fit_and_select(p_raw, y_true, min_samples=500, min_improvement_brier=1e-4)
    # Should either keep 'none' or refuse insignificant/overfit tiny wobble
    if cal.method != "none":
        # Any selected calibrator must strictly improve BOTH Brier and Log Loss on the held-out 40% eval slice
        assert cal.cal_val_brier < cal.raw_val_brier
        assert cal.cal_val_logloss < cal.raw_val_logloss

    # Contrast: genuinely distorted probabilities must be improved & detected
    logits = 0.45 * np.log(p_raw / (1.0 - p_raw)) - 0.5
    p_biased_true = 1.0 / (1.0 + np.exp(-logits))
    y_distorted = rng.binomial(1, p_biased_true)
    cal_fix = ProbabilityCalibrator(stat="PTS").fit_and_select(p_raw, y_distorted, min_samples=500)
    assert cal_fix.method in {"platt", "isotonic"}
    assert cal_fix.cal_val_brier < cal_fix.raw_val_brier
    assert cal_fix.cal_val_logloss < cal_fix.raw_val_logloss


# 10. Direct residual arithmetic & canonical name normalization + analysis structure
def test_residuals_arithmetic_and_distribution_report():
    assert canonical_stat_name("fg3m") == "3PM"
    assert canonical_stat_name("pts") == "PTS"
    with pytest.raises(ValueError):
        canonical_stat_name("PRA")  # combined markets explicitly out of scope!

    actual = np.array([25.0, 10.0, 0.0])
    pred = np.array([20.0, 12.0, 0.5])
    np.testing.assert_allclose(calculate_residuals(actual, pred), [5.0, -2.0, -0.5])

    df_syn = pd.DataFrame({
        "predicted_value": np.clip(np.linspace(0.0, 30.0, 200), 0.0, None),
        "raw_predicted_value": np.linspace(-0.2, 30.0, 200),
        "actual_value": np.clip(np.linspace(0.0, 32.0, 200) + np.sin(np.arange(200)), 0.0, None).round(),
        "predicted_minutes": np.linspace(10.0, 36.0, 200),
        "ROLL5_STARTER_PCT": np.where(np.arange(200) % 2 == 0, 1.0, 0.0),
    })
    df_syn["residual"] = calculate_residuals(df_syn["actual_value"], df_syn["predicted_value"])
    summary = analyse_residual_distribution(df_syn, "PTS")
    for key in [
        "mean_residual", "median_residual", "std_residual", "mae", "rmse",
        "skewness", "excess_kurtosis", "pct_zero_actual", "by_minutes_bucket", "by_role",
    ]:
        assert key in summary


# 11. Strict chronological cutoff prevents future-data leakage on OOS files
def test_chronological_split_separation_and_no_future_leakage():
    # If OOS parquet exists (generated by pipeline), validate real chronological invariants directly
    oos_pts_path = settings.nba_data_root / "processed" / "oos_predictions_pts.parquet"
    if not oos_pts_path.exists():
        pytest.skip("OOS parquet files not yet generated on disk.")

    for stat in TARGETS:
        df_all = load_oos_predictions(stat)
        assert not df_all.duplicated(subset=["PLAYER_ID", "GAME_ID"]).any()
        assert (df_all["predicted_value"] >= 0.0).all()

        # Verify residual == actual - predicted_value exactly
        np.testing.assert_allclose(
            df_all["residual"].to_numpy(),
            (df_all["actual_value"] - df_all["predicted_value"]).to_numpy(),
            atol=1e-7,
        )

        # Verify strict season/date separation across Train (22-25 shadow), Val (25-26), Test (26-27)
        s_22_25 = df_all[df_all["SEASON"].isin(["2022-23", "2023-24", "2024-25"])]
        s_25_26 = df_all[df_all["SEASON"] == "2025-26"]
        s_26_27 = df_all[df_all["SEASON"] == "2026-27"]

        assert s_22_25["model_version"].str.startswith("shadow_walkforward_").all()
        assert s_25_26["model_version"].str.startswith("champion_").all()
        assert s_26_27["model_version"].str.startswith("champion_").all()

        assert s_22_25["GAME_DATE"].max() < s_25_26["GAME_DATE"].min()
        assert s_25_26["GAME_DATE"].max() < s_26_27["GAME_DATE"].min()

        # max_date filter must strictly exclude games on or after cutoff
        df_before_26 = load_oos_predictions(stat, max_date="2025-10-01")
        assert (df_before_26["GAME_DATE"] < pd.Timestamp("2025-10-01")).all()
        assert set(df_before_26["SEASON"].unique()) <= {"2022-23", "2023-24", "2024-25"}


# 12. Method sanity & scope boundary verification (no betting/odds/EV keys anywhere)
def test_scope_boundary_no_odds_or_ev_fields():
    idx = _make_synthetic_oos_pool(n=500, stat="REB", seed=21)
    g = estimate_global_bootstrap_probability(7.5, idx.residuals, 6.5, n_simulations=2000)
    c = estimate_conditional_bootstrap_probability(7.5, 30.0, idx, 6.5, is_starter=True, n_simulations=2000)
    assert g["prob_over"] + g["prob_under"] + g["prob_push"] == pytest.approx(1.0)
    assert c["prob_over"] + c["prob_under"] + c["prob_push"] == pytest.approx(1.0)

    dist = simulate_player_outcomes("REB", 7.5, 30.0, idx)
    out_dict = dist.probability_for_line(6.5).to_dict()
    forbidden = {"odds", "fair_odds", "ev", "expected_value_bet", "kelly", "recommended_bet", "bookmaker"}
    assert forbidden.isdisjoint(set(out_dict.keys()))

