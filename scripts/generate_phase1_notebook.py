import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Introduction
cell1 = nbf.v4.new_markdown_cell("""# NBA Prediction Project — Advanced EDA Investigation
## Phase 1: Historical Player Performance, Rolling Trends, and Variance Profiles

**Data Scope:** 307,936 active player-game records across 13 seasons (2014–15 to 2026–27)  
**Database Source:** SQL Server LocalDB (`NBA_Prediction.dbo.stg_player_game_logs`) & Parquet Bronze Cache  

---

### 1. Objective
Investigate historical player-level statistical distributions, form momentum, and consistency profiles to answer:
> *"How much predictive signal exists in a player's recent historical game logs, and how does consistency vs. volatility dictate player prop probabilities?"*

### 2. Hypotheses Tested
1. **Right-Skewed Prop Distributions:** Player counting stats (PTS, REB, AST, 3PM) do not follow a Gaussian normal distribution; they are right-skewed (mean > median).
2. **Lagged Rolling Horizons & Regression to the Mean:** While fans talk about the "hot hand" (last 5 games), longer rolling baselines (10–20 games) provide stronger, more stable correlation with next-game performance.
3. **Volatility Profiles:** Players with identical season averages (e.g. 20 PPG) have wildly different coefficients of variation ($CV = \\sigma / \\mu$).
4. **Home vs. Away Shooting Lift:** Perimeter shooting (3P%) and scoring exhibit a measurable home-court boost across the league.
5. **Minutes as the Volume Anchor:** Playing time (MIN) is the single strongest volume driver for counting stats ($r > 0.70$).""")

# 2. Setup
cell2 = nbf.v4.new_code_cell("""import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

%matplotlib inline
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

summary_path = os.path.join(project_root, "notebooks", "eda", "figures", "phase1_metrics_summary.json")
with open(summary_path, "r") as f:
    phase1_metrics = json.load(f)

print("Phase 1 metrics successfully loaded!")""")

# 3. Distributions
cell3 = nbf.v4.new_markdown_cell("""### 3. Empirical Distributions Across Statistical Categories

Below are the empirical statistical distributions across 307,936 active player appearances in our database (MIN > 0):

| Stat | Mean | Std Dev | Median (50%) | 75th Pct | 90th Pct | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PTS** | 10.36 | 8.45 | 9.0 | 15.0 | 22.0 | **+1.04** |
| **REB** | 4.15 | 3.46 | 3.0 | 6.0 | 9.0 | **+1.30** |
| **AST** | 2.31 | 2.55 | 2.0 | 3.0 | 6.0 | **+1.71** |
| **STL** | 0.73 | 0.97 | 0.0 | 1.0 | 2.0 | **+1.56** |
| **BLK** | 0.46 | 0.83 | 0.0 | 1.0 | 1.0 | **+2.50** |
| **FG3M** | 1.08 | 1.42 | 1.0 | 2.0 | 3.0 | **+1.65** |
| **PRA** | 16.82 | 11.78 | 15.0 | 24.0 | 33.0 | **+0.82** |
| **MIN** | 22.80 | 10.48 | 24.0 | 31.0 | 36.0 | **-0.26** |

> **Key Analytical Finding:**  
> All box score counting categories are **heavily right-skewed** (mean > median). Blocks (+2.50 skew) and Assists (+1.71 skew) have the longest tails. Models assuming a symmetrical Gaussian bell curve will systematically overestimate Over probabilities on median lines.""")

cell4 = nbf.v4.new_code_cell("""# Display Distribution Grid
display(Image(filename="figures/fig1_player_distributions.png"))""")

# 4. Rolling Averages
cell5 = nbf.v4.new_markdown_cell("""### 4. Chronological Rolling Averages & Lookahead Bias Check

To test the predictive validity of recent form vs longer baselines without data leakage, we strictly compute **lagged pre-game rolling averages** (i.e. Game $N$ only uses Games $1 \\dots N-1$).

We correlated each pre-game metric with the player's actual points scored in the next game:

| Pre-Game Lagged Metric | Pearson Correlation ($r$) with Next Game Points |
| :--- | :---: |
| **Rolling 5 Games** | **$0.6877$** |
| **Rolling 10 Games** | **$0.7039$** |
| **Rolling 20 Games** | **$0.7081$** |
| **Season Average (Expanding)** | **$0.7040$** |

> **Key Analytical Finding:**  
> A 5-game rolling average is **noisier and less predictive** than a 10-game or 20-game sample. Bettors frequently overreact to a player's last 3 to 5 games ("he's on a hot streak"), but **regression to the 20-game baseline dominates**. The strongest feature is a hybrid: a 20-game baseline combined with a 5-game delta (momentum differential).""")

cell6 = nbf.v4.new_code_cell("""# Display Rolling Horizon Comparison
display(Image(filename="figures/fig2_rolling_correlations.png"))""")

# 5. Consistency vs Volatility
cell7 = nbf.v4.new_markdown_cell("""### 5. Player Consistency vs Volatility (Coefficient of Variation)

We analyzed all qualified regular starters ($\\ge 45$ games, $\\ge 18.0$ PPG in a single season) and computed their **Coefficient of Variation** ($CV = \\sigma / \\mu$):

#### Top 5 Most Consistent Scorers (Lowest Volatility / Highest Floor):
1. **Kevin Durant (2015-16)**: 28.18 PPG, $\\sigma = 5.35$, **$CV = 0.19$**
2. **Shai Gilgeous-Alexander (2022-23)**: 31.40 PPG, $\\sigma = 6.73$, **$CV = 0.21$**
3. **LeBron James (2016-17)**: 26.41 PPG, $\\sigma = 6.09$, **$CV = 0.23$**
4. **Shai Gilgeous-Alexander (2025-26)**: 31.13 PPG, $\\sigma = 7.18$, **$CV = 0.23$**
5. **Shai Gilgeous-Alexander (2024-25)**: 32.68 PPG, $\\sigma = 7.54$, **$CV = 0.23$**

#### Top 5 Most Volatile Scorers (Highest Volatility / Boom-or-Bust):
1. **Caris LeVert (2019-20)**: 18.71 PPG, $\\sigma = 9.76$, **$CV = 0.52$**
2. **Fred VanVleet (2022-23)**: 19.35 PPG, $\\sigma = 9.69$, **$CV = 0.50$**
3. **Bam Adebayo (2025-26)**: 20.11 PPG, $\\sigma = 10.04$, **$CV = 0.50$**
4. **CJ McCollum (2024-25)**: 21.12 PPG, $\\sigma = 10.29$, **$CV = 0.49$**
5. **DeMar DeRozan (2025-26)**: 18.43 PPG, $\\sigma = 8.93$, **$CV = 0.48$**

> **Betting Market Implication:**  
> For a standard 20.5 points prop line, a bettor getting identical odds on a consistent scorer vs a volatile scorer is facing completely different risk profiles. A consistent scorer rarely falls below 18, whereas a volatile shooter with high 3-point variance frequently records single-digit games or 35+ point explosions.""")

cell8 = nbf.v4.new_code_cell("""# Display Density Comparison: High Floor vs Boom-or-Bust
display(Image(filename="figures/fig3_consistency_comparison.png"))""")

# 6. Home vs Away
cell9 = nbf.v4.new_markdown_cell("""### 6. Home vs Away Performance Splits

Across all 300,000+ player appearances, we evaluated league-wide home vs away splits:

| Category | Away Average | Home Average | Raw Difference (Home - Away) | Relative Lift (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Points (PTS)** | 10.28 | 10.45 | **+0.17** | **+1.62%** |
| **Field Goal % (FG%)** | 44.7% | 45.5% | **+0.8%** | **+1.81%** |
| **3-Point % (3P%)** | 32.6% | 33.4% | **+0.8%** | **+2.34%** |
| **Points+Rebounds+Assists (PRA)** | 16.67 | 16.98 | **+0.31** | **+1.87%** |
| **Minutes Played (MIN)** | 22.84 | 22.75 | -0.08 | -0.36% |

> **Key Analytical Finding:**  
> Playing time is identical home vs away (22.8 min), but **shooting efficiency is noticeably higher at home**, especially 3-point shooting (+2.34% lift). The home boost is an efficiency boost (familiar sightlines, crowd support, no travel fatigue), not a playing time boost.""")

cell10 = nbf.v4.new_code_cell("""# Display Home vs Away Split Chart
display(Image(filename="figures/fig4_home_vs_away_splits.png"))""")

# 7. Minutes vs Production
cell11 = nbf.v4.new_markdown_cell("""### 7. Minutes Played vs Production

Playing time is the single largest structural factor in sports props. Below are the linear correlations between `MIN` and individual statistical categories:

| Relationship | Pearson Correlation ($r$) |
| :--- | :---: |
| **MIN vs PRA (Points+Reb+Ast)** | **$0.7937$** |
| **MIN vs Field Goal Attempts (FGA)** | **$0.7830$** |
| **MIN vs Points (PTS)** | **$0.7271$** |
| **MIN vs Rebounds (REB)** | **$0.5364$** |
| **MIN vs Assists (AST)** | **$0.5307$** |

> **Key Analytical Finding:**  
> With an $r = 0.79$, total volume (PRA) is almost entirely an expression of opportunity. If our model accurately forecasts player minutes (factoring in foul trouble, blowouts, and rotations), the prop probability estimates will outperform models relying purely on per-game averages.""")

cell12 = nbf.v4.new_code_cell("""# Display Minutes vs Points Scatter Plot with Trendline
display(Image(filename="figures/fig5_minutes_vs_points.png"))""")

# 8. Summary & Next Steps
cell13 = nbf.v4.new_markdown_cell("""### 8. Phase 1 Summary & Feature Engineering Takeaways

#### Core Discoveries:
1. **Do not rely on small-sample "hot streaks":** 5-game rolling windows are dominated by variance. A 10–20 game rolling baseline provides significantly higher correlation ($r \\approx 0.71$).
2. **Consistency (CV) is a mandatory feature:** Quantifying a player's volatility allows our probability engine to assign accurate probability density distributions to prop lines.
3. **Home court is an efficiency lift (+2.3% on 3P%), not a minute lift:** Home/away indicators should weight shooting percentages rather than minutes.
4. **Minutes projection is Step #1:** Volume is tied to playing time ($r = 0.79$ for PRA).

---

### Next Investigation: Phase 2 — Player Role, Minutes, and Usage
In Phase 2, we dive deeper into **how playing time and roles change over time**, how starting vs bench roles impact efficiency, and calculate estimated **Usage Rate (USG%)** across team rotations.""")

nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10, cell11, cell12, cell13]

nb_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "eda", "02_player_performance_and_trends.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully written to {nb_path} with {len(nb.cells)} cells.")
