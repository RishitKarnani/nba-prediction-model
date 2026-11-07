"""
scripts/build_processed_features.py
───────────────────────────────────
Industry-standard Offline Feature Store build script.
Precomputes all feature matrices and caches them to disk as Parquet files
in data/processed/ for instant sub-second model training and experimentation.
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.pipeline import build_player_feature_matrix, build_team_feature_matrix

def main():
    processed_dir = os.path.abspath("data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 60)
    print("NBA PREDICTION MODEL — OFFLINE FEATURE STORE BUILDER")
    print("=" * 60)
    print(f"Target directory: {processed_dir}\n")

    # 1. Build & Save Team Feature Matrix
    print("Building Team Feature Matrix (Win Probability / Spread)...")
    start_t = time.time()
    df_team = build_team_feature_matrix()
    team_path = os.path.join(processed_dir, "team_features.parquet")
    df_team.to_parquet(team_path, index=False)
    team_size = os.path.getsize(team_path) / (1024 * 1024)
    print(f"-> Saved: {team_path}")
    print(f"   Shape: {df_team.shape[0]:,} rows x {df_team.shape[1]} cols")
    print(f"   Size: {team_size:.2f} MB | Elapsed: {time.time() - start_t:.2f}s\n")

    # 2. Build & Save Player Feature Matrices for core prop stats
    core_stats = ["PTS", "REB", "AST"]
    for stat in core_stats:
        print(f"Building Player Feature Matrix for [{stat}]...")
        start_p = time.time()
        df_player = build_player_feature_matrix(stat)
        player_path = os.path.join(processed_dir, f"player_features_{stat.lower()}.parquet")
        df_player.to_parquet(player_path, index=False)
        player_size = os.path.getsize(player_path) / (1024 * 1024)
        print(f"-> Saved: {player_path}")
        print(f"   Shape: {df_player.shape[0]:,} rows x {df_player.shape[1]} cols")
        print(f"   Size: {player_size:.2f} MB | Elapsed: {time.time() - start_p:.2f}s\n")

    print("=" * 60)
    print("SUCCESS: All feature tables cached to data/processed/")
    print("Your model training scripts can now load them in under 0.1s!")
    print("=" * 60)

if __name__ == "__main__":
    main()

