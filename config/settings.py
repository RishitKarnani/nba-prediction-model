"""
config/settings.py
──────────────────
Centralised application configuration loaded from environment variables
(.env file). Every other module imports from here — never hardcode paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings, resolved from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # ── Data paths ────────────────────────────────────────────────────────────
    _DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
    _DEFAULT_ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

    nba_data_root: Path = _DEFAULT_DATA_ROOT
    artefacts_dir: Path = _DEFAULT_ARTEFACTS_DIR

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── Data loading ──────────────────────────────────────────────────────────
    # Comma-separated seasons, e.g. "2022-23,2023-24". Empty = load all.
    seasons_to_load: str = ""

    # ── Derived convenience properties ────────────────────────────────────────

    @property
    def player_logs_dir(self) -> Path:
        return self.nba_data_root / "raw" / "player_logs"

    @property
    def team_logs_dir(self) -> Path:
        return self.nba_data_root / "raw" / "team_logs"

    @property
    def sample_endpoints_dir(self) -> Path:
        primary = self.nba_data_root / "sample_api_endpoints"
        if primary.exists():
            return primary
        fallback = self.nba_data_root / "sample"
        if fallback.exists():
            return fallback
        return primary

    @property
    def seasons_filter(self) -> Optional[List[str]]:
        """Returns list of seasons to load, or None meaning load all."""
        if not self.seasons_to_load.strip():
            return None
        return [s.strip() for s in self.seasons_to_load.split(",") if s.strip()]

    # ── Feature engineering constants ─────────────────────────────────────────
    ROLLING_WINDOWS: List[int] = [5, 10, 20]

    # Minimum games required for a player to be included in model training
    MIN_GAMES_PLAYER: int = 10

    # Minimum minutes to count a game as "meaningful" for a player
    MIN_MINUTES_THRESHOLD: int = 5

    # ── Prediction constants ──────────────────────────────────────────────────
    # Number of bootstrap samples for distribution-based probability estimation
    N_BOOTSTRAP: int = 2000

    # Random seed for reproducibility
    RANDOM_SEED: int = 42


# Singleton instance — import this everywhere
settings = Settings()
