# NBA Player Prediction Project — Phase 2 Residual Analysis & Distribution Investigation

**Scope & Chronological Basis:**  
This report analyses strictly out-of-sample (OOS) historical prediction residuals (`Residual = Actual - Predicted`) generated without target-game leakage across seasons **`2022–23`, `2023–24`, and `2024–25`** (`n = 72,387` player-game observations per statistic). Predictions for `2022–23..2024–25` were produced via expanding walk-forward shadow LightGBM models trained solely on prior seasons (anchored by walk-forward out-of-sample `predicted_minutes` from `2019–20..2021–22`), while preserving the `2025–26` validation and `2026–27` test holdouts untouched for downstream calibration verification and backtesting. Raw model outputs were clamped at `>= 0.0` before residual calculation to enforce non-negative basketball statistics.

Generated datasets: `data/processed/oos_predictions_{pts,reb,ast,3pm,stl,blk,tov}.parquet` (`98,638` total rows each spanning `2022–23` through `2026–27`).  
Diagnostic plots: `artefacts/plots/probability/`.

---

## 1. Executive Findings Across All Seven Statistics

1. **None of the seven residual distributions is Normally distributed** (`Jarque-Bera p = 0.0` across all 7 stats; `is_approx_normal = False`).
2. **Systematic Positive Mean Bias from L1 (Median) Loss:** Because all seven LightGBM models optimize `regression_l1` (conditional median), their **median residuals cluster near `0.0`** (`-0.117` for `PTS`, `-0.048` for `REB`, and `-0.001` to `+0.001` for `AST`, `3PM`, `STL`, `BLK`, `TOV`), whereas **mean residuals are systematically positive** (`+0.155` to `+0.547`) due to right-skewed upper tails. Passing raw point predictions directly into a zero-mean Normal curve systematically underestimates true means and over-hit probabilities.
3. **Severe Zero-Inflation & Discrete Clumping in Rare Events:**  
   - **Blocks (`BLK`):** **`66.24%`** zero actual outcomes (`p25 = p50 = 0.0` residual, skewness `1.735`, excess kurtosis `5.681`).  
   - **Steals (`STL`):** **`49.87%`** zero actual outcomes (skewness `1.096`, excess kurtosis `1.902`).  
   - **Three-Pointers (`3PM`):** **`40.09%`** zero actual outcomes (skewness `1.152`, excess kurtosis `2.724`).  
   - **Turnovers (`TOV`):** **`33.75%`** zero actual outcomes (skewness `0.954`, excess kurtosis `1.711`).  
   - Raw LightGBM L1 regression occasionally outputs slight negative values on zero-prone bench projections (`STL: 6.78%`, `BLK: 4.30%`, `3PM: 1.01%`, `TOV: 0.60%`, `AST: 0.23%`), confirming the necessity of pre-clamping `max(0.0, raw_pred)`.
4. **Strong Volume & Minutes Heteroscedasticity:** Absolute error magnitude (`|Residual|`) correlates positively with both predicted stat magnitude (`r = +0.191` to `+0.381`) and predicted playing time (`r = +0.101` to `+0.243`). Standard deviation of `PTS` errors expands from **`4.45` (`<15m`)** to **`7.75` (`32+m`)**; `AST` expands from **`1.29` (`<15m`)** to **`2.40` (`32+m`)**.

---

## 2. Summary Residual Metrics (2022–23 to 2024–25 OOS, `n = 72,387`)

| Stat | Mean Res | Median Res | Std Dev | MAE | RMSE | % Zero Actual | Skewness | Excess Kurtosis | % > 2 SD | Domain Large Error % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PTS** | `+0.5469` | `-0.1170` | `6.1156` | `4.6848` | `6.1400` | `6.68%` | `+0.5692` | `1.3438` | `5.22%` | `9.73%` (`>=10`) |
| **REB** | `+0.2704` | `-0.0484` | `2.5390` | `1.9177` | `2.5533` | `6.84%` | `+0.7796` | `1.8201` | `5.30%` | `10.23%` (`>=4`) |
| **AST** | `+0.2702` | `+0.0005` | `1.8506` | `1.3468` | `1.8702` | `20.37%` | `+0.8773` | `2.5327` | `6.10%` | `10.01%` (`>=3`) |
| **3PM** | `+0.2284` | `-0.0010` | `1.2765` | `0.8733` | `1.2968` | `40.09%` | `+1.1524` | `2.7241` | `6.27%` | `10.44%` (`>=2`) |
| **STL** | `+0.3001` | `0.0000` | `0.9506` | `0.6717` | `0.9969` | `49.87%` | `+1.0961` | `1.9022` | `8.84%` | `5.89%` (`>=2`) |
| **BLK** | `+0.3176` | `0.0000` | `0.7717` | `0.4491` | `0.8345` | `66.24%` | `+1.7350` | `5.6812` | `7.60%` | `5.11%` (`>=2`) |
| **TOV** | `+0.1546` | `-0.0005` | `1.2040` | `0.8908` | `1.2139` | `33.75%` | `+0.9543` | `1.7108` | `5.30%` | `9.05%` (`>=2`) |

### Residual Quantiles (`Actual - Predicted`)

| Stat | P01 | P05 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PTS** | `-12.750` | `-8.269` | `-6.316` | `-3.495` | `-0.117` | `+4.074` | `+8.420` | `+11.321` | `+17.800` |
| **REB** | `-4.859` | `-3.224` | `-2.524` | `-1.397` | `-0.048` | `+1.675` | `+3.551` | `+4.875` | `+7.840` |
| **AST** | `-3.743` | `-2.243` | `-1.686` | `-0.964` | `0.000` | `+1.070` | `+2.675` | `+3.716` | `+5.990` |
| **3PM** | `-2.112` | `-1.240` | `-1.011` | `-0.699` | `-0.001` | `+0.969` | `+1.986` | `+2.860` | `+4.139` |
| **STL** | `-1.006` | `-0.997` | `-0.937` | `-0.125` | `0.000` | `+0.993` | `+1.661` | `+2.001` | `+3.034` |
| **BLK** | `-1.011` | `-0.657` | `-0.096` | `0.000` | `0.000` | `+0.953` | `+1.000` | `+2.000` | `+3.000` |
| **TOV** | `-2.108` | `-1.209` | `-1.003` | `-0.856` | `-0.001` | `+0.948` | `+1.895` | `+2.328` | `+3.922` |

---

## 3. Relationship with Predicted Value, Minutes & Player Role

### Heteroscedasticity Correlations

| Stat | `corr(pred, |residual|)` | `corr(pred, residual)` | `corr(pred_min, |residual|)` | `corr(pred_min, residual)` |
|---|---:|---:|---:|---:|
| **PTS** | `+0.2524` | `-0.0222` | `+0.2261` | `-0.0192` |
| **REB** | `+0.2871` | `+0.0169` | `+0.1010` | `+0.0046` |
| **AST** | `+0.3650` | `+0.0013` | `+0.2428` | `+0.0083` |
| **3PM** | `+0.3812` | `-0.0047` | `+0.2154` | `+0.0320` |
| **STL** | `+0.1913` | `-0.1771` | `+0.1632` | `-0.1041` |
| **BLK** | `+0.3174` | `-0.0350` | `+0.1061` | `+0.0545` |
| **TOV** | `+0.2857` | `+0.0066` | `+0.2329` | `+0.0496` |

### Breakdown by Expected Minutes Bucket (`<15m`, `15–25m`, `25–32m`, `32+m`)

- **PTS Std Dev:** `<15m`: `4.454` (`n=8,681`) → `15-25m`: `5.365` (`n=28,025`) → `25-32m`: `6.392` (`n=21,586`) → `32+m`: `7.746` (`n=14,095`).
- **REB Std Dev:** `<15m`: `2.198` → `15-25m`: `2.425` → `25-32m`: `2.655` → `32+m`: `2.765`.
- **AST Std Dev:** `<15m`: `1.288` → `15-25m`: `1.584` → `25-32m`: `1.948` → `32+m`: `2.401`.
- **3PM Std Dev:** `<15m`: `0.888` → `15-25m`: `1.130` → `25-32m`: `1.363` → `32+m`: `1.578`.
- **STL Std Dev:** `<15m`: `0.730` → `15-25m`: `0.868` → `25-32m`: `1.006` → `32+m`: `1.098`.
- **BLK Std Dev:** `<15m`: `0.651` → `15-25m`: `0.717` → `25-32m`: `0.839` → `32+m`: `0.829`.
- **TOV Std Dev:** `<15m`: `0.867` → `15-25m`: `1.026` → `25-32m`: `1.265` → `32+m`: `1.554`.

Notice `STL` mean residual shifts substantially with expected minutes and predicted tier: low-minute players (`<15m`) have `mean_res = +0.388` (median prediction collapses near `0.0`), while high-minute starters (`32+m`) drop to `mean_res = +0.168`. Conditioning residuals locally removes this bias distortion.

### Breakdown by Starter vs. Bench Role (`Starters n = 40,240`, `Bench n = 32,147`)

| Stat | Starters MAE | Bench MAE | Starters Std Dev | Bench Std Dev | Starters % Zero | Bench % Zero |
|---|---:|---:|---:|---:|---:|---:|
| **PTS** | `5.3236` | `3.8850` | `6.8214` | `5.0935` | `2.02%` | `12.52%` |
| **REB** | `2.0492` | `1.7532` | `2.6831` | `2.3462` | `3.16%` | `11.44%` |
| **AST** | `1.5673` | `1.0707` | `2.0891` | `1.4992` | `10.91%` | `32.21%` |
| **3PM** | `1.0208` | `0.6887` | `1.4262` | `1.0581` | `29.60%` | `53.21%` |
| **STL** | `0.7625` | `0.5581` | `1.0326` | `0.8229` | `42.33%` | `59.31%` |
| **BLK** | `0.5109` | `0.3718` | `0.8276` | `0.6932` | `61.87%` | `71.72%` |
| **TOV** | `1.0295` | `0.7172` | `1.3565` | `0.9756` | `22.70%` | `47.58%` |

---

## 4. Distributional Implications for Probability Estimation

1. **Why Gaussian Assumptions (Method A) Break Down:**  
   - Assumes symmetry (`skewness = 0`), fixed global variance (ignoring `4.45` vs `7.75` `PTS` spread), and positive continuous support around zero.  
   - Even with mean bias adjustment (`pred + mean_residual`) and continuity correction, Normal tails poorly reflect multi-modal integer spikes at `0` (`66.24%` in `BLK`, `49.87%` in `STL`, `40.09%` in `3PM`) and over-spread bench rows while under-spreading stars.
2. **Why Global Bootstrap (Method B) Remains Suboptimal:**  
   - Samples empirical shape faithfully on average, including non-zero spikes, **but** mixes `+8` residual swings from `34`-minute starters into `12`-minute bench projections (and vice versa), creating miscalibrated variance by player role and expected volume.
3. **Why Conditional Residual Bootstrap (Method C) Fits the Empirical Structure:**  
   - Conditioning on `(predicted_value, predicted_minutes, starter_role)` matches each player to historical games with identical median-prediction bias, volume variance scale, minutes context, and zero-inflation propensity—capturing discrete clumping (`BLK`/`STL`/`3PM`) and heteroscedasticity (`PTS`/`REB`/`AST`) organically without parametric distortions.

---

## 5. Generated Visualisations

All visualisations are saved under `artefacts/plots/probability/`:
- [`residual_histograms_normal_overlay.png`](artefacts/plots/probability/residual_histograms_normal_overlay.png)
- [`predicted_vs_actual_scatter.png`](artefacts/plots/probability/predicted_vs_actual_scatter.png)
- [`residuals_vs_predicted.png`](artefacts/plots/probability/residuals_vs_predicted.png)
- [`residuals_by_minutes_bucket.png`](artefacts/plots/probability/residuals_by_minutes_bucket.png)
- [`error_distribution_by_role.png`](artefacts/plots/probability/error_distribution_by_role.png)

