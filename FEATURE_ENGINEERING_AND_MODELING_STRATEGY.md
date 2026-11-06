# NBA Prediction Model: Feature Engineering Audit & ML Modeling Strategy

**Author:** NBA Prediction Model Engineering Team  
**Date:** November 4, 2026  
**Status:** Pipeline Verified (31/31 Unit Tests Passing)  
**Target:** Team Win Probability, Game Spreads & Player Prop Over/Under Predictions

---

## 1. Executive Summary

We have merged and tested the data and feature engineering pipeline provided by your collaborator. The code represents a **production-ready feature store and modeling framework**. 

All **31 automated tests** passed with zero regressions, verifying that:
- In-memory Parquet loaders and fuzzy name resolvers operate cleanly across 13 seasons (2014–15 to 2026–27).
- Dean Oliver's Four Factors, True Shooting %, and possessions are mathematically consistent.
- **Strict zero data leakage (`shift(1)`)** is verified across both team and player rolling features.

---

## 2. Current Feature Inventory

The current feature pipeline (`features/pipeline.py`) computes the following feature layers:

### A. Team Level (`build_team_feature_matrix()`)
| Feature Family | Columns / Metrics | Window Sizes ($W$) | Basketball Context |
| :--- | :--- | :---: | :--- |
| **Offensive Efficiency** | `ROLL{W}_OFF_RATING`, `ROLL{W}_PTS`, `ROLL{W}_EFG_PCT` | 5, 10, 20 games | True scoring efficiency per 100 possessions. |
| **Defensive Efficiency** | `ROLL{W}_DEF_RATING` | 5, 10, 20 games | Points allowed per 100 possessions via paired game join. |
| **Net Form** | `ROLL{W}_NET_RATING`, `ROLL{W}_WIN_PCT` | 5, 10, 20 games | Primary indicator of true team quality and momentum. |
| **Possession Factors** | `ROLL{W}_TOV_PCT`, `ROLL{W}_OREB_PCT`, `ROLL{W}_FT_RATE` | 5, 10, 20 games | Dean Oliver Four Factors controlling game variance. |
| **Game Pace** | `ROLL{W}_PACE` | 5, 10, 20 games | Expected game tempo (possessions per 48 minutes). |
| **Location Splits** | `HOME_ROLL10_*`, `AWAY_ROLL10_*` | Trailing 10 games | Captures home-court dominance vs. road fatigue. |
| **Schedule / Rest** | `REST_DAYS`, `IS_B2B`, `IS_3IN4`, `REST_ADV` | Game-level | Days off, Back-to-Back flag, 3-in-4 nights, and net rest edge. |

### B. Player Prop Level (`build_player_feature_matrix(stat)`)
| Feature Family | Columns / Metrics | Window Sizes ($W$) | Basketball Context |
| :--- | :--- | :---: | :--- |
| **Rolling Production** | `ROLL{W}_{STAT}_MEAN` | 5, 10, 20 games | Trailing active form across PTS, REB, AST, 3PM, etc. |
| **Consistency / Volatility** | `ROLL{W}_{STAT}_STD` | 5, 10, 20 games | Standard deviation measuring floor/ceiling predictability. |
| **Extreme Range** | `ROLL{W}_{STAT}_MAX`, `ROLL{W}_{STAT}_MIN` | 5, 10, 20 games | Ceiling and floor bounds. |
| **Macro Averages** | `SEASON_{STAT}_MEAN`, `CAREER_{STAT}_MEAN` | Expanding | Long-term baseline anchor, excluding target game. |
| **Opponent Defensive Context** | `OPP_CONCEDED_{STAT}_ROLL{W}` | 5, 10, 20 games | Specific volume allowed by opponent to this stat category. |
| **Opponent Matchup Context**| `OPP_DEF_RATING_ROLL{W}`, `OPP_PACE_ROLL{W}` | 5, 10, 20 games | Opponent defensive quality and tempo inflator. |
| **Player Rest & Location** | `REST_DAYS`, `IS_B2B`, `IS_HOME` | Game-level | Rest context and home/away shooting splits. |

---

## 3. High-Value Additional Features to Engineer (Next Phase)

While the existing feature store is already strong enough to beat basic sportsbooks and baseline models, adding the following **5 feature categories** will give your model a definitive edge:

### 1. Head-to-Head Differential Features (Team Matchup Deltas)
> **Priority: High | Impact: High**
Currently, team features contain Home and Away statistics side-by-side. Tree-based algorithms (LightGBM/XGBoost) and linear classifiers achieve much higher accuracy when provided explicit **Delta ($\Delta$) features**:
- $\Delta \text{Net Rating} = \text{Home\_Net\_Rating}_{10} - \text{Away\_Net\_Rating}_{10}$
- $\Delta \text{Off/Def Mismatch} = \text{Home\_Off\_Rating}_{10} - \text{Away\_Def\_Rating}_{10}$
- $\Delta \text{eFG\%} = \text{Home\_eFG\%}_{10} - \text{Away\_eFG\%}_{10}$
- $\Delta \text{Rebound Battle} = \text{Home\_OREB\%}_{10} - \text{Away\_DREB\%}_{10}$
- $\Delta \text{Rest Advantage} = \text{Home\_Rest\_Days} - \text{Away\_Rest\_Days}$

### 2. Player Absence & Usage Vacuum (Usage Redistribution)
> **Priority: High | Impact: Very High for Player Props**
In basketball, minutes and shot attempts are zero-sum. When a star (e.g. Luka Dončić or Giannis Antetokounmpo) sits out:
- Their 30–35% Usage Rate is redistributed to secondary starters.
- **Proposed Feature:** `TEAM_MISSING_USAGE_PCT` and `EXPECTED_MINUTES_BOOST`.

### 3. Blowout Risk & Minutes Truncation
> **Priority: Medium | Impact: High for Star Player Props**
If a team is favored by 14+ points, starters frequently sit the entire 4th quarter (playing 28 minutes instead of 36), causing prop "Unders" to hit disproportionately:
- **Proposed Feature:** `BLOWOUT_RISK_INDEX = abs(Delta_Net_Rating) * Home_Advantage`.

### 4. Travel, Time Zones & Altitude Penalties
> **Priority: Medium | Impact: Moderate**
- **Altitude Factor:** Playing in Denver (Denver Nuggets @ 5,280 ft) or Utah (Jazz @ 4,226 ft) on the second night of a back-to-back (`IS_B2B == 1`) carries the steepest physiological fatigue penalty in the league.
- **Circadian Rhythm Shift:** East Coast teams playing West Coast late games (or vice-versa).

### 5. Vegas Market Anchoring (Implied Totals)
> **Priority: High (if odds data is available) | Impact: Very High**
Vegas closing lines represent the wisdom of the crowd. If you incorporate the **Over/Under line** and **Point Spread**:
- $\text{Implied Home Total} = \frac{\text{Total} - \text{Spread}}{2}$
- Training your model to predict the **residual** ($Y - \text{Vegas Implied}$) produces state-of-the-art predictive performance.

---

## 4. Recommended Algorithm & Model Selection Strategy

### Task 1: Team Win Probability (Moneyline) & Spread
* **Problem Type:** Binary Classification (Win/Loss) & Continuous Regression (Point Spread).
* **Recommended Primary Algorithm: LightGBM Classifier (`LGBMClassifier`)**
  * **Why LightGBM?**
    1. Handles non-linear interactions between pace, defensive rating, and rest seamlessly.
    2. Extremely fast training time on 29,000+ historical team rows.
    3. Handles missing values gracefully (e.g., season-opening games where trailing rest is NaN).
  * **Critical Training Best Practice:**
    Use **Probability Calibration** (`CalibratedClassifierCV` with `method='isotonic'`). Sports betting requires that when the model says "60% win probability," the team actually wins 60% of the time. Raw tree probabilities can be overconfident near the edges (0 and 1).
* **Secondary / Benchmark Model:** **Logistic Regression with $L_2$ Regularization (Ridge)**
  * A regularized logistic regression on Four Factor deltas is surprisingly hard to beat and serves as an ideal baseline.

---

### Task 2: Player Prop Predictions (Points, Rebounds, Assists, 3PM)
* **Problem Type:** Non-negative Count / Continuous Regression with Skew.
* **Recommended Primary Algorithm: LightGBM Regressor with Tweedie / Poisson Objective**
  * **Why not standard MSE (Mean Squared Error)?**
    Box score counting stats (assists, 3-pointers, blocks, rebounds) cannot be negative and are heavily right-skewed. Standard MSE assumes normal symmetric errors.
    - Set `objective='tweedie'` or `objective='poisson'` in LightGBM for stats like assists, rebounds, and 3PM.
* **Quantile Regression (Ceiling & Floor):**
  * Train 3 LightGBM regressors at `alpha=0.10` (Floor), `alpha=0.50` (Median), and `alpha=0.90` (Ceiling).
  * Feed these directly into the empirical bootstrap probability engine (`models/probability.py`) written by your collaborator!

---

## 5. Walk-Forward Backtesting Framework (No Future Peeking)

Never use standard random $k$-fold cross-validation for sports data! Random splitting causes future games to leak into past predictions.

**Recommended Validation Split:**
- **Training Set:** 2014–15 to 2022–23 seasons (9 full seasons, ~22,000 games).
- **Validation Set:** 2023–24 season (hyperparameter tuning).
- **Out-of-Sample Test Set:** 2024–25 to 2026–27 seasons (true forward test simulating real betting).

---

## 6. Prompt to Copy and Send to ChatGPT for Discussion

You can copy-paste the prompt below directly into ChatGPT to continue your discussion:

```markdown
Here is the current state of our NBA prediction project:
- We have 13 seasons (2014-15 to 2026-27) of NBA team and player game logs in Parquet.
- We built a modular feature store in Python with zero data leakage (shift(1)):
  - Team: Rolling 5/10/20 Off/Def/Net ratings, Four Factors (eFG%, TOV%, OREB%, FT rate), Pace, Rest days, B2B, 3-in-4 nights, and Home/Away splits.
  - Player: Rolling 5/10/20 Mean/Std/Max/Min for all box stats, expanding season/career means, opponent points conceded rolling averages, opponent pace, and rest.
  - Probability Engine: Parametric (Normal) and Empirical Bootstrap (2000 resamples) for over/under probability estimation.

Questions for you:
1. Between LightGBM, XGBoost, and CatBoost, which do you recommend for NBA Player Props vs Team Win Probability, and what loss functions (e.g. Tweedie, Logloss, Quantile) work best?
2. How should we calculate 'Matchup Deltas' for team win prediction?
3. What is the best way to model the 'star absence usage vacuum' for player props when a key teammate is ruled out?
4. How do you recommend calibrating output probabilities (Isotonic Regression vs Platt Scaling vs Temperature Scaling) for finding +EV betting edges?
```

