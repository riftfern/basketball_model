"""SQLAlchemy ORM models for the basketball betting model."""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean,
    ForeignKey, Enum as SQLEnum, JSON, Numeric, UniqueConstraint, Index,
    Text
)
from sqlalchemy.orm import relationship, declarative_base

from ..config.constants import PropType

Base = declarative_base()


class Team(Base):
    """NBA team information."""
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    nba_api_id = Column(Integer, unique=True, nullable=False)
    abbreviation = Column(String(3), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    city = Column(String(50))
    conference = Column(String(10))
    division = Column(String(20))

    players = relationship("Player", back_populates="team")
    season_stats = relationship("TeamSeasonStats", back_populates="team")


class Player(Base):
    """NBA player information."""
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    nba_api_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"))
    position = Column(String(10))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    team = relationship("Team", back_populates="players")
    game_logs = relationship("PlayerGameLog", back_populates="player")
    projections = relationship("Projection", back_populates="player")
    bets = relationship("Bet", back_populates="player")


class Game(Base):
    """NBA game information."""
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)
    nba_api_id = Column(String(20), unique=True, nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    season = Column(String(10), nullable=False)  # e.g., "2024-25"
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    is_completed = Column(Boolean, default=False)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    player_logs = relationship("PlayerGameLog", back_populates="game")


class PlayerGameLog(Base):
    """Individual game statistics for a player."""
    __tablename__ = "player_game_logs"
    __table_args__ = (
        UniqueConstraint('player_id', 'game_id', name='uq_player_game'),
        Index('ix_player_game_logs_player_date', 'player_id', 'game_date'),
    )

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=True)  # Can be null for imports
    game_date = Column(Date, nullable=False, index=True)

    # Playing time
    minutes = Column(Float)

    # Core stats
    points = Column(Integer)
    rebounds = Column(Integer)
    offensive_rebounds = Column(Integer)
    defensive_rebounds = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)

    # Shooting
    fgm = Column(Integer)
    fga = Column(Integer)
    fg3m = Column(Integer)
    fg3a = Column(Integer)
    ftm = Column(Integer)
    fta = Column(Integer)

    # Context
    is_home = Column(Boolean)
    opponent_team_id = Column(Integer, ForeignKey("teams.id"))
    rest_days = Column(Integer)  # Days since last game
    is_back_to_back = Column(Boolean)

    player = relationship("Player", back_populates="game_logs")
    game = relationship("Game", back_populates="player_logs")

    @property
    def pra(self) -> int:
        """Points + Rebounds + Assists."""
        return (self.points or 0) + (self.rebounds or 0) + (self.assists or 0)


class TeamSeasonStats(Base):
    """Aggregated team stats for a season."""
    __tablename__ = "team_season_stats"
    __table_args__ = (
        UniqueConstraint('team_id', 'season', name='uq_team_season'),
    )

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season = Column(String(10), nullable=False)
    games_played = Column(Integer)

    # Offensive stats
    off_rating = Column(Float)  # Points per 100 possessions
    pace = Column(Float)        # Possessions per game
    pts_per_game = Column(Float)

    # Defensive stats
    def_rating = Column(Float)
    opp_pts_per_game = Column(Float)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    team = relationship("Team", back_populates="season_stats")


class PropLine(Base):
    """Sportsbook prop lines."""
    __tablename__ = "prop_lines"
    __table_args__ = (
        Index('ix_prop_lines_lookup', 'player_id', 'game_id', 'prop_type'),
    )

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    sportsbook = Column(String(50), nullable=False)  # e.g., "draftkings", "fanduel"
    prop_type = Column(SQLEnum(PropType), nullable=False)

    line = Column(Numeric(5, 1), nullable=False)  # e.g., 25.5
    over_odds = Column(Integer)   # American odds, e.g., -110
    under_odds = Column(Integer)

    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_current = Column(Boolean, default=True)


class Projection(Base):
    """Model-generated projections."""
    __tablename__ = "projections"
    __table_args__ = (
        UniqueConstraint('player_id', 'game_id', 'prop_type', 'model_version',
                        name='uq_projection'),
    )

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    prop_type = Column(SQLEnum(PropType), nullable=False)
    model_version = Column(String(20), nullable=False, default="v1")

    # Point estimates
    predicted_value = Column(Float, nullable=False)
    predicted_std = Column(Float)

    # Distribution parameters
    distribution_type = Column(String(20))  # poisson, neg_binom, normal
    distribution_params = Column(JSON)      # {"n": 5, "p": 0.3} etc.

    # Pre-computed probabilities at common thresholds
    threshold_probs = Column(JSON)  # {"15.5": 0.92, "20.5": 0.75, ...}

    created_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player", back_populates="projections")


class Bet(Base):
    """Bet tracking."""
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    sportsbook = Column(String(50))

    prop_type = Column(SQLEnum(PropType), nullable=False)
    line = Column(Numeric(5, 1), nullable=False)
    is_over = Column(Boolean, nullable=False)  # True = over, False = under

    odds = Column(Integer, nullable=False)     # American odds at time of bet
    stake = Column(Numeric(10, 2), nullable=False)

    # Model edge at time of bet
    model_prob = Column(Float)          # Model's probability
    implied_prob = Column(Float)        # Implied prob from odds
    expected_value = Column(Float)      # EV percentage

    # Outcome
    actual_value = Column(Float)        # Actual stat result
    outcome = Column(String(10))        # "win", "loss", "push", "pending"
    profit_loss = Column(Numeric(10, 2))

    placed_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime)
    notes = Column(Text)

    player = relationship("Player", back_populates="bets")
