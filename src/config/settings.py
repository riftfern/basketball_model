"""Application settings and configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Paths
    base_dir: Path = Field(
        default=Path(__file__).parent.parent.parent,
        description="Base directory of the project"
    )

    @property
    def db_path(self) -> Path:
        return self.base_dir / "data" / "basketball.db"

    # NBA API
    nba_api_rate_limit: int = Field(default=20, description="Requests per minute")

    # The Odds API
    odds_api_key: str = Field(default="", description="API key for the-odds-api.com")
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # Model settings
    min_games_for_prediction: int = Field(default=10, description="Minimum games needed")
    recency_weight: float = Field(default=0.15, description="Exponential decay weight")

    # Betting settings
    min_edge: float = Field(default=0.02, description="Minimum edge to bet (2%)")
    min_ev: float = Field(default=0.03, description="Minimum EV to bet (3%)")
    kelly_fraction: float = Field(default=0.25, description="Fraction of Kelly to use")
    max_bet_pct: float = Field(default=0.05, description="Max bet as % of bankroll")

    class Config:
        env_file = ".env"
        env_prefix = "NBA_"


settings = Settings()
