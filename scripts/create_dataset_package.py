"""Package Parquet datasets and DATA_DICTIONARY.md into a single zip file for sharing."""

import os
import glob
import zipfile

def main():
    zip_filename = "nba_prediction_dataset.zip"
    print(f"Packaging dataset into {zip_filename}...")
    
    # Files to include
    team_logs = sorted(glob.glob("data/raw/team_logs/*.parquet"))
    player_logs = sorted(glob.glob("data/raw/player_logs/*.parquet"))
    sample_files = sorted(glob.glob("data/sample/*.parquet"))
    
    print(f"Found {len(team_logs)} team log files")
    print(f"Found {len(player_logs)} player log files")
    print(f"Found {len(sample_files)} sample endpoint files")
    
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # 1. Add DATA_DICTIONARY.md at the root
        if os.path.exists("DATA_DICTIONARY.md"):
            zipf.write("DATA_DICTIONARY.md", arcname="nba_prediction_dataset/DATA_DICTIONARY.md")
            print("Added DATA_DICTIONARY.md")
            
        # 2. Add raw/player_logs/*.parquet
        for p_file in player_logs:
            arcname = os.path.join("nba_prediction_dataset", "raw", "player_logs", os.path.basename(p_file))
            zipf.write(p_file, arcname=arcname)
            
        # 3. Add raw/team_logs/*.parquet
        for t_file in team_logs:
            arcname = os.path.join("nba_prediction_dataset", "raw", "team_logs", os.path.basename(t_file))
            zipf.write(t_file, arcname=arcname)
            
        # 4. Add sample/*.parquet as sample_api_endpoints
        for s_file in sample_files:
            arcname = os.path.join("nba_prediction_dataset", "sample_api_endpoints", os.path.basename(s_file))
            zipf.write(s_file, arcname=arcname)
            
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\nSuccessfully created '{zip_filename}'! Total size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()

