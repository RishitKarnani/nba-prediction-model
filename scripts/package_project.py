"""Package the complete runnable NBA Prediction Model project into a single ZIP archive.

Excludes local secrets (.env), virtual environments (.venv), git metadata (.git),
caches (__pycache__, .pytest_cache), old ZIP archives, local SQL backups (*.bak),
and legacy collaboration staging directories (friend_submission).
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "friend_submission",
}

EXCLUDED_FILES = {
    ".env",
}

EXCLUDED_EXTENSIONS = {
    ".zip",
    ".bak",
    ".pyc",
    ".pyo",
}

QUICKSTART_TEXT = """NBA Player Performance & Probability Engine (v1) — Friend Setup Guide
=======================================================================

1. Create a Python virtual environment (Python 3.10+ recommended) and install packages:
   python -m venv .venv
   # Windows:
   .venv\\Scripts\\activate
   # macOS/Linux:
   # source .venv/bin/activate
   pip install -r requirements.txt

2. (Optional) Copy .env.example to .env if needed (local defaults work without .env).

3. Run a player performance & threshold probability query right away:
   python scripts/query_player.py --player "LeBron James" --opponent GSW --away --date 2026-11-01 --line PTS:20.5,25.0,25.5 --line AST:6.5

4. Or call from Python:
   from models.inference import PlayerProbabilityEngine
   engine = PlayerProbabilityEngine()
   res = engine.predict_player(
       player="Stephen Curry",
       opponent="LAL",
       is_home=1,
       lines_by_stat={"PTS": [20.5, 25.5, 30.5], "3PM": [3.5, 4.5]},
   )

5. Verify the installation with the automated test suite (54 tests):
   pytest tests/ -v

6. Key Reports:
   - PROBABILITY_ENGINE_FINAL_REPORT.md
   - PROBABILITY_RESIDUAL_ANALYSIS_REPORT.md
   - MINUTES_MODEL_AND_RETRAINING_SHOWDOWN_REPORT.md
   - MODEL_EVALUATION_AND_ALGORITHM_SHOWDOWN_REPORT.md
   - DATA_DICTIONARY.md
"""


def should_skip(rel_path: Path, out_zip_name: str, runnable_only: bool) -> bool:
    """Return True if rel_path should not be packaged into the friend ZIP."""
    parts = set(rel_path.parts)
    if parts & EXCLUDED_DIRS:
        return True
    if rel_path.name in EXCLUDED_FILES or rel_path.name == out_zip_name:
        return True
    if rel_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    # Empty legacy database folder is only needed for SQL Server .bak restoration
    if rel_path.parts and rel_path.parts[0] == "database":
        return True

    if runnable_only and len(rel_path.parts) >= 3 and rel_path.parts[:2] == ("data", "processed"):
        if not (rel_path.name.startswith("oos_predictions_") and rel_path.suffix == ".parquet"):
            return True

    return False


def build_zip(output_path: Path, runnable_only: bool = False) -> tuple[int, int]:
    """Create ZIP archive rooted under `NBA_Prediction_Model/`."""
    prefix = "NBA_Prediction_Model"
    file_count = 0
    total_raw_bytes = 0

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(f"{prefix}/FRIEND_QUICKSTART.txt", QUICKSTART_TEXT)

        for current_root, dirnames, filenames in os.walk(ROOT_DIR):
            cur_path = Path(current_root)
            rel_dir = cur_path.relative_to(ROOT_DIR)

            # Prune excluded directories in-place so os.walk never enters .venv or .git
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS and d != "database")

            for fname in sorted(filenames):
                full_path = cur_path / fname
                rel_path = rel_dir / fname if str(rel_dir) != "." else Path(fname)
                if should_skip(rel_path, output_path.name, runnable_only=runnable_only):
                    continue

                arcname = f"{prefix}/{rel_path.as_posix()}"
                zf.write(full_path, arcname=arcname)
                file_count += 1
                total_raw_bytes += full_path.stat().st_size

    return file_count, total_raw_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Package NBA Prediction Model into a friend-ready ZIP.")
    parser.add_argument(
        "--output",
        default="NBA_Prediction_Model_v1_full.zip",
        help="Output ZIP filename at repository root.",
    )
    parser.add_argument(
        "--runnable-only",
        action="store_true",
        help="Omit large intermediate feature/split parquets while keeping raw + oos + artefacts.",
    )
    args = parser.parse_args()

    out_path = (ROOT_DIR / args.output).resolve()
    count, raw_bytes = build_zip(out_path, runnable_only=args.runnable_only)
    zip_mb = out_path.stat().st_size / (1024 * 1024)
    raw_mb = raw_bytes / (1024 * 1024)
    print(f"Created: {out_path}")
    print(f"Files packaged: {count} (+ FRIEND_QUICKSTART.txt)")
    print(f"Uncompressed size: {raw_mb:.2f} MB -> Compressed ZIP size: {zip_mb:.2f} MB")


if __name__ == "__main__":
    main()

