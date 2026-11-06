"""Phase 1 EDA Script: Historical Player Performance, Rolling Trends, and Variance Analysis.

Computes empirical distributions, rolling recency correlations, consistency profiles,
home/away splits, and minutes production curves.
Saves figures to notebooks/eda/figures/ and generates summary metrics.
"""

import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.connection import get_engine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "eda", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


def run_phase1_eda():
    print("=" * 70)
    print("PHASE 1 EDA: HISTORICAL PLAYER PERFORMANCE AND TRENDS")
    print("=" * 70)

    # 1. Health check SQL Server
    engine = get_engine()
    with engine.connect() as conn:
        db_count = conn.execute(text("SELECT COUNT(*) FROM dbo.stg_player_game_logs")).scalar()
        print(f"Verified SQL Server dbo.stg_player_game_logs: {db_count:,} records.")

    # 2. Fast load from Parquet cache (0.34s)
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "player_logs")
    start_time = time.time()
    print(f"Loading player game logs from Parquet cache ({raw_path})...")
    df = pd.read_parquet(raw_path)
    print(f"Loaded {len(df):,} player-game logs in {round(time.time() - start_time, 2)}s.")

    # Filter to active appearances (MIN > 0)
    df = df[df["MIN"] > 0].copy()
    print(f"Active player game logs (MIN > 0): {len(df):,} rows.")

    # Format dates and composite metrics
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)

    results = {}

    # -------------------------------------------------------------
    # 1. Performance Distributions
    # -------------------------------------------------------------
    print("\n--- 1. Computing Descriptive Distributions ---")
    cat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "PRA", "MIN"]
    dist_stats = df[cat_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T
    dist_stats["skewness"] = df[cat_cols].skew()
    results["distributions"] = dist_stats[["mean", "std", "50%", "75%", "90%", "99%", "skewness"]].to_dict()

    print(dist_stats[["mean", "std", "50%", "75%", "90%", "skewness"]].round(2).to_string())

    # Plot Distribution Grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_cats = ["PTS", "REB", "AST", "FG3M", "PRA", "MIN"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, col in enumerate(plot_cats):
        ax = axes[idx // 3, idx % 3]
        sns.histplot(df[col], bins=35, kde=True, ax=ax, color=palette[idx])
        ax.set_title(f"{col} Distribution (Skew: {df[col].skew():.2f})", fontsize=12, fontweight="bold")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="black", linestyle="-", label=f"Median: {df[col].median():.1f}")
        ax.legend(loc="upper right")

    plt.suptitle("NBA Player Performance Distributions (2014-2026, 300k+ Records)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_player_distributions.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  Saved figure: {fig1_path}")

    # -------------------------------------------------------------
    # 2. Rolling Averages & Lookahead Bias Check
    # -------------------------------------------------------------
    print("\n--- 2. Computing Chronological Rolling Averages (Strict Zero Lookahead) ---")
    df = df.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)

    grouped = df.groupby(["PLAYER_ID", "SEASON"])
    # Strictly shift(1) to only use prior games
    df["ROLL_PTS_5"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    df["ROLL_PTS_10"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    df["ROLL_PTS_20"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(20, min_periods=10).mean())
    df["SEASON_PTS_AVG"] = grouped["PTS"].transform(lambda x: x.shift(1).expanding(min_periods=5).mean())

    valid_rolls = df.dropna(subset=["ROLL_PTS_5", "ROLL_PTS_10", "ROLL_PTS_20", "SEASON_PTS_AVG", "PTS"])
    corr_next_pts = {
        "corr_rolling_5_vs_next_pts": round(float(valid_rolls["ROLL_PTS_5"].corr(valid_rolls["PTS"])), 4),
        "corr_rolling_10_vs_next_pts": round(float(valid_rolls["ROLL_PTS_10"].corr(valid_rolls["PTS"])), 4),
        "corr_rolling_20_vs_next_pts": round(float(valid_rolls["ROLL_PTS_20"].corr(valid_rolls["PTS"])), 4),
        "corr_season_avg_vs_next_pts": round(float(valid_rolls["SEASON_PTS_AVG"].corr(valid_rolls["PTS"])), 4),
    }
    results["rolling_correlations"] = corr_next_pts
    print("  Correlations with Next Game PTS:")
    for k, v in corr_next_pts.items():
        print(f"    {k}: {v:.4f}")

    # Plot Rolling Horizon Comparison
    plt.figure(figsize=(10, 5))
    horizons = ["Rolling 5 Games", "Rolling 10 Games", "Rolling 20 Games", "Season Average (Expanding)"]
    corrs = [corr_next_pts["corr_rolling_5_vs_next_pts"], corr_next_pts["corr_rolling_10_vs_next_pts"],
             corr_next_pts["corr_rolling_20_vs_next_pts"], corr_next_pts["corr_season_avg_vs_next_pts"]]
    bars = plt.bar(horizons, corrs, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"], width=0.5)
    plt.title("Correlation Between Pre-Game Lagged Metrics and Next Game Scoring", fontsize=13, fontweight="bold")
    plt.ylabel("Pearson Correlation (r)")
    plt.ylim(0.4, 0.75)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_rolling_correlations.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 3. Consistency vs Volatility Profiles (Coefficient of Variation)
    # -------------------------------------------------------------
    print("\n--- 3. Analyzing Player Consistency vs Volatility ---")
    player_season_stats = df.groupby(["PLAYER_ID", "PLAYER_NAME", "SEASON"]).agg(
        games=("GAME_ID", "count"),
        mean_pts=("PTS", "mean"),
        std_pts=("PTS", "std"),
        mean_min=("MIN", "mean"),
    ).reset_index()

    qualified_scorers = player_season_stats[(player_season_stats["games"] >= 45) & (player_season_stats["mean_pts"] >= 18.0)].copy()
    qualified_scorers["cv"] = qualified_scorers["std_pts"] / qualified_scorers["mean_pts"]
    qualified_scorers = qualified_scorers.sort_values("cv")

    top_consistent = qualified_scorers.head(5)[["PLAYER_NAME", "SEASON", "games", "mean_pts", "std_pts", "cv"]]
    top_volatile = qualified_scorers.tail(5)[["PLAYER_NAME", "SEASON", "games", "mean_pts", "std_pts", "cv"]]

    print("\n  Top 5 Most Consistent Scorers (Lowest Coefficient of Variation):")
    print(top_consistent.round(2).to_string(index=False))

    print("\n  Top 5 Most Volatile Scorers (Highest Coefficient of Variation):")
    print(top_volatile.round(2).to_string(index=False))

    # Pick two comparable players around ~20-22 PPG with different volatility
    fig, ax = plt.subplots(figsize=(11, 5))
    consistent_example = top_consistent.iloc[0]
    volatile_example = top_volatile.iloc[-1]

    df_p1 = df[(df["PLAYER_NAME"] == consistent_example["PLAYER_NAME"]) & (df["SEASON"] == consistent_example["SEASON"])]["PTS"]
    df_p2 = df[(df["PLAYER_NAME"] == volatile_example["PLAYER_NAME"]) & (df["SEASON"] == volatile_example["SEASON"])]["PTS"]

    sns.kdeplot(df_p1, ax=ax, fill=True, label=f"Consistent: {consistent_example['PLAYER_NAME']} ({consistent_example['SEASON']}) - Mean: {df_p1.mean():.1f}, Std: {df_p1.std():.1f}, CV: {consistent_example['cv']:.2f}", color="navy")
    sns.kdeplot(df_p2, ax=ax, fill=True, label=f"Volatile: {volatile_example['PLAYER_NAME']} ({volatile_example['SEASON']}) - Mean: {df_p2.mean():.1f}, Std: {df_p2.std():.1f}, CV: {volatile_example['cv']:.2f}", color="crimson")
    ax.set_title("Probability Density: Consistent (High Floor) vs Volatile (Boom-or-Bust) Scorer", fontsize=13, fontweight="bold")
    ax.set_xlabel("Points Scored in a Game")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_consistency_comparison.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 4. Home vs Away Performance Splits
    # -------------------------------------------------------------
    print("\n--- 4. Computing Home vs Away Performance Splits ---")
    home_away = df.groupby("IS_HOME").agg(
        pts=("PTS", "mean"),
        fg_pct=("FG_PCT", "mean"),
        fg3_pct=("FG3_PCT", "mean"),
        pra=("PRA", "mean"),
        min=("MIN", "mean"),
    ).T
    home_away.columns = ["Away", "Home"]
    home_away["Diff (Home - Away)"] = home_away["Home"] - home_away["Away"]
    home_away["Pct_Lift"] = (home_away["Diff (Home - Away)"] / home_away["Away"]) * 100
    print(home_away.round(3).to_string())
    results["home_away_splits"] = home_away.to_dict()

    # Plot Home vs Away lift
    plt.figure(figsize=(10, 5))
    metrics = ["PTS", "PRA", "MIN"]
    home_vals = [home_away.loc["pts", "Home"], home_away.loc["pra", "Home"], home_away.loc["min", "Home"]]
    away_vals = [home_away.loc["pts", "Away"], home_away.loc["pra", "Away"], home_away.loc["min", "Away"]]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, away_vals, width, label="Away", color="#7f7f7f")
    plt.bar(x + width/2, home_vals, width, label="Home", color="#1f77b4")
    plt.xticks(x, ["Points", "PRA", "Minutes"])
    plt.title("League-Wide Performance: Home vs Away (2014-2026)", fontsize=13, fontweight="bold")
    plt.ylabel("Per-Game Average")
    plt.legend()
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_home_vs_away_splits.png")
    plt.savefig(fig4_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 5. Minutes Played vs Production
    # -------------------------------------------------------------
    print("\n--- 5. Evaluating Minutes vs Production ---")
    min_corrs = {
        "corr_min_pts": round(float(df["MIN"].corr(df["PTS"])), 4),
        "corr_min_reb": round(float(df["MIN"].corr(df["REB"])), 4),
        "corr_min_ast": round(float(df["MIN"].corr(df["AST"])), 4),
        "corr_min_pra": round(float(df["MIN"].corr(df["PRA"])), 4),
        "corr_min_fga": round(float(df["MIN"].corr(df["FGA"])), 4),
    }
    results["minutes_correlations"] = min_corrs
    print("  Correlations with Playing Time (MIN):")
    for k, v in min_corrs.items():
        print(f"    {k}: {v:.4f}")

    sample_for_scatter = df.sample(n=10000, random_state=42)
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=sample_for_scatter, x="MIN", y="PTS",
        scatter_kws={"alpha": 0.15, "color": "#1f77b4", "s": 15},
        line_kws={"color": "red", "linewidth": 2}
    )
    plt.title(f"Minutes Played vs Points Scored (r = {min_corrs['corr_min_pts']:.3f})", fontsize=13, fontweight="bold")
    plt.xlabel("Minutes Played in Game")
    plt.ylabel("Points Scored")
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "fig5_minutes_vs_points.png")
    plt.savefig(fig5_path, dpi=200)
    plt.close()

    # Save summary JSON
    summary_file = os.path.join(FIGURES_DIR, "phase1_metrics_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"PHASE 1 ANALYSIS COMPLETE IN {round(time.time() - start_time, 2)}s")
    print(f"Metrics saved to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_eda()
