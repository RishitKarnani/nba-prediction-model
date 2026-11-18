# NBA Tactical Action Simulation & Game Margin Prediction Engine
## Final League Simulation Report & Model Integration

---

### Executive Summary

We have fully implemented and executed the end-to-end tactical simulation and game prediction system across all **30 NBA franchises**, simulating **8,700 live games** (10 Home and 10 Away games for all 870 directed league fixtures).

The entire pipeline is active, tested, and integrated with the project:
1. **LightGBM Model-Predicted Minutes**: Every player's playing time on the 15-man active rosters was generated strictly by the trained LightGBM minutes model (`artefacts/models/minutes/minutes_model.joblib`), normalized to the regulation 240-minute invariant ($\sum \text{MIN} = 240.0$).
2. **Tactical Action Matrix Synergy**: Each team's 5 Offensive Actions and 5 Defensive Coverages were mapped into tactical domain archetypes and evaluated through an Expected Points Per 100 Possessions (PPP) counter matrix.
3. **Non-Zero Weight Adjustment ($w_k > 0$)**: 17,400 individual tactical weights across all matchups were dynamically adjusted in proportion to expected win probability against each opponent. **Zero weights were recorded at 0.00%** (minimum observed weight was $11.63\%$, sum strictly $100.0\%$).
4. **Game Outcome & Margin Predictor**: Operational via `models/team_inference.py` and CLI `scripts/predict_game.py`, delivering predicted winners, win probabilities, final scores, point margins, and optimal tactical gameplans.

---

## 1. 8,700-Game League Standings & Performance Power Rankings

Below are the aggregated results from simulating all 30 NBA teams across 580 games each (29 opponents $\times$ 20 games per opponent = 8,700 total games):

| Rank | Team | Record (W–L) | Win % | PPG | Opp PPG | Net Margin | Primary Tactical Driver |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Houston Rockets** (`HOU`) | 494–86 | **85.2%** | 125.4 | 116.3 | **+9.2** | Physical Switch 1–5 & Short-Roll Hub Playmaking |
| **2** | **Denver Nuggets** (`DEN`) | 491–89 | **84.7%** | 126.7 | 118.1 | **+8.6** | Elbow DHO Two-Man Game & Backdoor Cut Synergy |
| **3** | **Boston Celtics** (`BOS`) | 477–103 | **82.2%** | 122.9 | 115.2 | **+7.7** | 5-Out Drive & Kick / Inverted PnR Spacing |
| **4** | **San Antonio Spurs** (`SAS`) | 468–112 | **80.7%** | 125.9 | 119.0 | **+6.9** | Elite Rim Deterrence Drop & Vertical Pop PnR |
| **5** | **Los Angeles Lakers** (`LAL`) | 426–154 | **73.4%** | 121.5 | 116.4 | **+5.1** | Spain PnR Baseline Cuts & Switch 1–4 Length |
| **6** | **Cleveland Cavaliers** (`CLE`) | 423–157 | **72.9%** | 124.5 | 119.2 | **+5.2** | Two-Big Rim Deterrence & Surgical Spread PnR |
| **7** | **Oklahoma City Thunder** (`OKC`) | 409–171 | **70.5%** | 122.7 | 118.2 | **+4.5** | 5-Out Clear-Side Iso & Swarming Switch Defense |
| **8** | **Detroit Pistons** (`DET`) | 405–175 | **69.8%** | 121.0 | 116.7 | **+4.3** | High PnR Short-Roll Kickouts & POA Denial |
| **9** | **Toronto Raptors** (`TOR`) | 396–184 | **68.3%** | 122.2 | 118.1 | **+4.0** | Positionless Switch Everything & Delay 5-Out DHOs |
| **10** | **New York Knicks** (`NYK`) | 393–187 | **67.8%** | 116.8 | 113.6 | **+3.2** | Top Rim Protection Drop & Corner Glass Crashes |
| **11** | **Atlanta Hawks** (`ATL`) | 356–224 | **61.4%** | 123.7 | 121.0 | **+2.7** | Spain PnR Backscreens & High Spread PnR |
| **12** | **Miami Heat** (`MIA`) | 348–232 | **60.0%** | 123.9 | 121.0 | **+2.8** | 2–3 Matchup Zone & Zoom Elbow DHO Movement |
| **13** | **Indiana Pacers** (`IND`) | 308–272 | **53.1%** | 120.1 | 119.4 | **+0.7** | Ultra-Fast Transition & Random Motion Flow |
| **14** | **Portland Trail Blazers** (`POR`) | 306–274 | **52.8%** | 120.0 | 119.8 | **+0.3** | Spread PnR Creation & Athletic Wing Switches |
| **15** | **LA Clippers** (`LAC`) | 289–291 | **49.8%** | 116.4 | 116.3 | **+0.1** | Mid-Post Clearouts & Heavy Gap Dig-Downs |
| **16** | **Phoenix Suns** (`PHX`) | 289–291 | **49.8%** | 116.5 | 116.8 | **-0.2** | Midrange Pull-Up Reads & Spain/Stack Sets |
| **17** | **Golden State Warriors** (`GSW`) | 272–308 | **46.9%** | 116.6 | 117.2 | **-0.6** | Step-Up Slip Screens & Show/Recover Defense |
| **18** | **Charlotte Hornets** (`CHA`) | 265–315 | **45.7%** | 115.8 | 116.7 | **-0.9** | Early Drag Screens & Deep 30ft Ball Screens |
| **19** | **Minnesota Timberwolves** (`MIN`) | 237–343 | **40.9%** | 120.2 | 121.6 | **-1.5** | Deep Drop Rim Funneling & Lob Threats |
| **20** | **New Orleans Pelicans** (`NOP`) | 219–361 | **37.8%** | 118.1 | 120.7 | **-2.6** | Point-Forward Bull Drives & Switch 1–5 Length |
| **21** | **Dallas Mavericks** (`DAL`) | 214–366 | **36.9%** | 119.4 | 121.9 | **-2.5** | Double-Drag Screens & Weak-Side Low-Man Tags |
| **22** | **Orlando Magic** (`ORL`) | 192–388 | **33.1%** | 117.6 | 121.5 | **-3.9** | Downhill Wing Drives & Switch 1–5 Length |
| **23** | **Sacramento Kings** (`SAC`) | 191–389 | **32.9%** | 113.4 | 117.1 | **-3.7** | High-Post Zoom DHOs & High Show Pressure |
| **24** | **Philadelphia 76ers** (`PHI`) | 177–403 | **30.5%** | 117.0 | 120.9 | **-3.9** | Elbow Post Isolation & Paint Drop Defense |
| **25** | **Utah Jazz** (`UTA`) | 143–437 | **24.7%** | 118.5 | 124.0 | **-5.5** | 5-Out Motion & Perimeter Pick-and-Pop |
| **26** | **Milwaukee Bucks** (`MIL`) | 133–447 | **22.9%** | 112.1 | 118.0 | **-5.9** | Deep Drop Defense & Post-Up Hub Sets |
| **27** | **Memphis Grizzlies** (`MEM`) | 131–449 | **22.6%** | 115.3 | 121.4 | **-6.1** | High-Pace Transition & Delay 5-Out Motion |
| **28** | **Washington Wizards** (`WAS`) | 109–471 | **18.8%** | 114.5 | 122.1 | **-7.6** | Fast Transition Pushes & Drop Scheme |
| **29** | **Chicago Bulls** (`CHI`) | 105–475 | **18.1%** | 117.4 | 125.4 | **-7.9** | Chicago Pindown DHO & POA Pressure |
| **30** | **Brooklyn Nets** (`BKN`) | 34–546 | **5.9%** | 105.8 | 118.2 | **-12.4** | Horns Entries & Switch 1–4 Defense |

---

## 2. Tactical Weight Optimization Verification

> [!IMPORTANT]
> **Strict Compliance Check:**
> - **Total Weights Evaluated:** $17,400$
> - **Zero-Weight Occurrences:** $0$ ($0.00\%$)
> - **Minimum Observed Weight:** $11.63\%$ (exceeds the theoretical floor $\epsilon = 4.0\%$)
> - **Maximum Observed Weight:** $28.42\%$
> - **Sum of Action Weights:** Strictly $100.00\%$ per team per contest.

### Example Tactical Rebalancing: `DAL @ BOS`
When Dallas plays Boston at TD Garden:
- Boston's default defense features *Switch 1–5* and *"X-Out" Scramble Rotations*.
- In response, Dallas dynamically increases its weight on **Horns Out / Double-Drag Screens** from $20.0\% \to \mathbf{25.3\%}$ (to create multi-level screen chaos against switches) and **Lob / Vertical Spacer Dive Plays** to $\mathbf{21.1\%}$.
- Dallas reduces **Isolation Clear-Outs** from $20.0\% \to \mathbf{15.4\%}$ (because Boston's switchable wing length heavily suppresses isolation efficiency).
- Boston encounters Dallas's rim-protecting drop and weak-side tags, adjusting **Guard-Guard Inverted PnR** to $\mathbf{20.3\%}$ and **Ghost Screens** to $\mathbf{20.3\%}$ to drag Dallas's perimeter defenders into immediate cross-matches.

---

## 3. How to Query the Team Prediction Model

### 3.1 Via Command-Line Interface (`scripts/predict_game.py`)
Run predictions directly from the terminal specifying `--home` and `--away` by abbreviation or full franchise name:

```bash
# Example 1: Boston Celtics hosting Dallas Mavericks
/opt/anaconda3/bin/python scripts/predict_game.py --home BOS --away DAL

# Example 2: Los Angeles Lakers hosting Golden State Warriors
/opt/anaconda3/bin/python scripts/predict_game.py --home "Los Angeles Lakers" --away "Golden State Warriors"
```

#### Sample Terminal Output (`DAL @ BOS`):
```text
============================================================================
  NBA GAME PREDICTION & TACTICAL ANALYSIS: DAL @ BOS
  Dallas Mavericks at Boston Celtics
============================================================================

[GAME OUTCOME PREDICTION]
  Predicted Winner    : Boston Celtics (BOS)
  Win Probability     : BOS 100.0% | DAL 0.0%
  Predicted Score     : BOS 130.2  -  DAL 115.9
  Predicted Margin    : +14.3 (BOS)
  Implied Spread Line : BOS -14.3

[TACTICAL GAMEPLAN: BOSTON CELTICS (HOME)]
  Top Offensive Counter : Guard-Guard Inverted PnR
  Top Defensive Counter : "X-Out" Scramble Rotations
  Adjusted Offensive Weights (Non-Zero):
    - 5-Out Drive & Kick                     :  20.0% | ■■■■■■■■■
    - Guard-Guard Inverted PnR               :  20.3% | ■■■■■■■■■■
    - Post-Up Mismatch Hunting               :  19.5% | ■■■■■■■■■
    - Ghost Screens (slipping before contact) :  20.3% | ■■■■■■■■■■
    - High-Volume Above-the-Break 3s         :  20.0% | ■■■■■■■■■

[TACTICAL GAMEPLAN: DALLAS MAVERICKS (AWAY)]
  Top Offensive Counter : Horns Out / Double-Drag Screens
  Top Defensive Counter : Weak-Side Low-Man Tags
  Adjusted Offensive Weights (Non-Zero):
    - High Spread PnR (1-on-1 Mismatch)      :  18.8% | ■■■■■■■■■
    - Lob / Vertical Spacer Dive Plays       :  21.1% | ■■■■■■■■■■
    - Isolation Clear-Outs                   :  15.4% | ■■■■■■■
    - Corner Skip-Pass Spot-Ups              :  19.3% | ■■■■■■■■■
    - Horns Out / Double-Drag Screens        :  25.3% | ■■■■■■■■■■■■

[MODEL-PREDICTED LINEUP MINUTES (LIGHTGBM)]
  BOS Top Rotation:
    * Jaylen Brown           [Starter  ] : 26.4 min (11.0%)
    * Jayson Tatum           [Starter  ] : 25.9 min (10.8%)
    * Derrick White          [Starter  ] : 23.3 min (9.7%)
    * Payton Pritchard       [Starter  ] : 22.0 min (9.2%)
    * Anfernee Simons        [Starter  ] : 18.6 min (7.8%)
  DAL Top Rotation:
    * Cooper Flagg           [Starter  ] : 24.9 min (10.3%)
    * P.J. Washington        [Starter  ] : 23.7 min (9.9%)
    * Naji Marshall          [Starter  ] : 20.6 min (8.6%)
    * Ryan Nembhard          [Starter  ] : 19.3 min (8.1%)
    * Max Christie           [Starter  ] : 19.3 min (8.0%)
============================================================================
```

### 3.2 Via Python API (`models/team_inference.py`)
```python
from models.team_inference import TeamPredictionEngine

engine = TeamPredictionEngine()
result = engine.predict_game(home_team="LAL", away_team="GSW")

print("Winner:", result["prediction"]["predicted_winner"])
print("Win Prob:", result["prediction"]["home_win_probability"])
print("Predicted Score:", result["prediction"]["predicted_score"])
print("Scoring Margin:", result["prediction"]["predicted_margin"])
print("Top Home Counter:", result["tactical_gameplan"]["home_team"]["best_offensive_action"])
```

---

## 4. Verification & Automated Test Suite

A test suite was added to [`tests/test_tactical_team_prediction.py`](file:///Users/arushdahra/Downloads/NBA_Prediction_Model/tests/test_tactical_team_prediction.py), passing with 100% success rate:
- `test_tactical_registry_integrity`: Validates 30 teams $\times$ 5 actions $\times$ valid archetypes.
- `test_weight_optimizer_strictly_non_zero`: Validates the $w_k \ge \epsilon$ invariant and exact $1.00$ sum.
- `test_team_prediction_engine`: Tests end-to-end game prediction generation and margin formatting.
- `test_team_abbreviation_resolver`: Tests robust string resolution across city names, franchise nicknames, and abbreviations.
