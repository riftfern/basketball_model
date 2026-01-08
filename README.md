# NBA Player Props Betting Model

A quantitative platform for detecting mispriced player prop markets in the NBA. Integrates historical performance data, real-time sportsbook odds, and contextual game factors into a unified prediction system that identifies positive expected value opportunities.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Type Checked](https://img.shields.io/badge/Type%20Checked-mypy-blue.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-orange.svg)

## Problem Statement

Sports betting markets are inefficient. Sportsbooks set player prop lines based on public perception, liability management, and competitor pricing—not purely on statistical expectation. This creates exploitable edges for models that:

1. Accurately estimate probability distributions for player performance
2. Account for contextual factors the market underweights (opponent strength, pace, rest)
3. Compare model probabilities against devigged market odds to isolate true edge
4. Size positions optimally given edge magnitude and uncertainty

This system addresses each component through a modular pipeline that separates data ingestion, statistical modeling, odds processing, and decision logic.

## Key Features

### Statistical Analysis
- **Negative Binomial Model**: Handles overdispersion (variance > mean) common in basketball statistics
- **Context-Aware Projections**: Adjusts for opponent defensive rating, game pace, and situational factors
- **Recency Weighting**: Exponential decay prioritizes recent performance
- **Multi-Prop Support**: Points, rebounds, assists, steals, blocks, threes, turnovers, and combo props (PRA)

### Betting Analytics
- **Expected Value Calculation**: `EV = P(win) × (odds - 1) - P(lose)`
- **Edge Detection**: Model probability vs. devigged fair probability comparison
- **Kelly Criterion**: Optimal bankroll allocation with fractional Kelly (default 25%)
- **Multi-Book Comparison**: Identifies best available lines across sportsbooks

### Odds Processing
- **Vig Removal**: Multiplicative, power, and worst-case devigging methods
- **American/Decimal Conversion**: Full odds format support
- **Live Odds Integration**: Real-time data from FanDuel, DraftKings, BetMGM, and more

## Architecture

```
src/
├── cli.py                    # Typer CLI interface
├── config/
│   ├── settings.py           # Pydantic configuration
│   └── constants.py          # Enums and thresholds
├── db/
│   ├── session.py            # SQLAlchemy session management
│   └── models.py             # ORM models
├── data/
│   ├── sources/
│   │   └── nba_api_source.py # NBA API with rate limiting
│   └── collectors/
│       └── player_stats.py   # Data collection logic
├── models/
│   ├── base.py               # Abstract base classes
│   └── statistical/
│       └── negative_binomial.py  # Prediction model
├── betting/
│   ├── ev_calculator.py      # EV and edge calculations
│   └── kelly.py              # Bet sizing
├── odds/
│   ├── odds_api.py           # The Odds API client
│   └── devig.py              # Vig removal utilities
└── sheets/
    └── sheets_tool.py        # Google Sheets integration
```

## Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    NBA API      │     │  The Odds API   │     │  Google Sheets  │
│  (Historical)   │     │  (Real-time)    │     │    (Output)     │
└────────┬────────┘     └────────┬────────┘     └────────▲────────┘
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────────────────────────────────┐         │
│              Rate-Limited Clients           │         │
│         (20 req/min NBA, async HTTP)        │         │
└────────┬───────────────────────┬────────────┘         │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐             │
│   SQLAlchemy    │     │  In-Memory      │             │
│   (Persistent)  │◄───►│  (Session)      │             │
└────────┬────────┘     └────────┬────────┘             │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────────────────────────────────┐         │
│           Prediction Engine                 │         │
│  (Neg. Binomial + Context Adjustments)      │         │
└────────────────────┬────────────────────────┘         │
                     │                                   │
                     ▼                                   │
┌─────────────────────────────────────────────┐         │
│            Edge Detection                   │         │
│   (Devig → Compare → Kelly → Filter)        │─────────┘
└─────────────────────────────────────────────┘
```

**Data Sources:**
- **NBA API**: Player game logs, team statistics, roster data. Rate-limited to 20 requests/minute with exponential backoff.
- **The Odds API**: Real-time lines from 10+ sportsbooks. Normalized to consistent format across books with different conventions.
- **SQLite**: Persistent storage for historical data, predictions, and audit trail.

**Key Integration Challenges Solved:**
- Inconsistent player name formats across APIs (resolved via fuzzy matching + ID mapping)
- Sportsbook odds format variance (American, decimal, fractional → unified internal representation)
- API rate limits during bulk operations (queue-based throttling with configurable concurrency)
- Data freshness for live betting (TTL-based caching, forced refresh option)

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.11+, NumPy, Pandas, SciPy |
| **Database** | SQLAlchemy, SQLite |
| **APIs** | NBA API, The Odds API, Google Sheets API |
| **CLI** | Typer, Rich |
| **HTTP** | HTTPX (async-capable) |
| **Config** | Pydantic, python-dotenv |
| **Quality** | mypy, Ruff, pytest |

## Installation

```bash
# Clone the repository
git clone https://github.com/riftfern/basketball_model.git
cd basketball_model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Configuration

```env
# Required for live odds
NBA_ODDS_API_KEY=your_api_key_here

# Model parameters (optional - shown with defaults)
NBA_MIN_EDGE=0.02           # Minimum edge threshold (2%)
NBA_MIN_EV=0.03             # Minimum EV threshold (3%)
NBA_KELLY_FRACTION=0.25     # Quarter Kelly
NBA_MAX_BET_PCT=0.05        # Max 5% of bankroll per bet
NBA_MIN_GAMES_FOR_PREDICTION=10
NBA_RECENCY_WEIGHT=0.15     # Exponential decay factor
```

## Usage

### Initial Setup

```bash
# Initialize database
nba init

# Sync NBA data
nba sync-teams
nba sync-players
nba sync-team-stats

# Backfill historical game logs
nba backfill --seasons 2024-25
```

### Player Analysis

```bash
# View recent stats for a player
nba stats "LeBron James"

# Generate projection with specific line
nba project "LeBron James" --prop points --line 24.5 \
  --over-odds -110 --under-odds -110
```

**Example Output:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           LeBron James - Points Projection       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Predicted Mean:     26.4                         │
│ Standard Deviation: 8.2                          │
│ P(Over 24.5):       65.2%                        │
│ Fair Odds:          -187 / +153                  │
│ Edge (Over):        +3.2%                        │
│ EV (Over):          +5.1%                        │
│ Kelly Sizing:       2.1% of bankroll             │
└─────────────────────────────────────────────────┘
```

### Live Edge Detection

```bash
# Scan for +EV opportunities with live odds
nba live-edges --output-csv picks.csv

# View current sportsbook lines
nba fetch-odds --prop points
```

### Google Sheets Integration

```bash
# Authenticate with Google
nba sheets auth

# Upload picks to spreadsheet
nba sheets upload <spreadsheet_id> --csv-path picks.csv
```

## Statistical Methodology

### Negative Binomial Distribution

The model uses negative binomial distributions rather than Poisson because basketball statistics exhibit **overdispersion** (variance exceeds mean). Parameter estimation uses method of moments:

```python
mean = weighted_average(historical_data)
variance = weighted_variance(historical_data)
p = mean / variance
n = mean * p / (1 - p)
```

### Contextual Adjustments

| Factor | Adjustment Method |
|--------|-------------------|
| Opponent Defense | Scale by defensive rating ratio (full for points, partial for other stats) |
| Game Pace | Square-root scaling of pace ratio |
| Home/Away | +2% boost for home games |
| Back-to-Back | -3% adjustment |
| Extended Rest | +1% for 3+ days rest |
| Recency | Exponential decay weighting (λ = 0.15) |

### Edge Detection Algorithm

1. Fit negative binomial to weighted historical data
2. Apply contextual adjustments for upcoming matchup
3. Calculate threshold probability: `P(X > line)`
4. Devig sportsbook odds to get fair probability
5. Compute edge: `model_prob - fair_prob`
6. Compute EV: `model_prob × decimal_odds - 1`
7. Apply Kelly criterion for bet sizing
8. Flag as +EV if edge ≥ 2% AND EV ≥ 3%

## Database Schema

The SQLite database stores:

- **Players**: NBA player metadata and team affiliations
- **Teams**: Team info with pace, offensive/defensive ratings
- **GameLogs**: Per-game player statistics (100+ fields)
- **PropLines**: Sportsbook odds with timestamps
- **Predictions**: Model outputs with distribution parameters

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run type checking
mypy src/

# Run linter
ruff check src/

# Run tests
pytest tests/
```

## Design Decisions

### Why Negative Binomial Over Poisson?

Poisson distributions assume variance equals mean. Basketball statistics consistently exhibit overdispersion—a player averaging 25 points might have variance of 40+ due to game script variability, foul trouble, and blowouts. Negative binomial's extra parameter captures this, producing more accurate tail probabilities critical for betting thresholds.

### Why Method of Moments Over MLE?

For this sample size (10-82 games per player), method of moments provides comparable accuracy to maximum likelihood estimation with significantly lower computational cost. This matters when generating projections for 400+ players across 7 prop types in real-time.

### Why SQLite Over PostgreSQL?

The system runs as a single-user CLI tool. SQLite provides ACID compliance, zero configuration, and sub-millisecond query latency for the access patterns here (primarily single-player lookups and batch inserts). The schema is designed to migrate to PostgreSQL if multi-user access becomes necessary.

### Why Fractional Kelly?

Full Kelly criterion maximizes long-term growth rate but produces high volatility. Quarter Kelly (default) sacrifices ~25% of expected growth for ~50% reduction in drawdown variance. The fraction is configurable based on risk tolerance.

### Separation of Concerns

The architecture strictly separates:
- **Data layer**: Ingestion and storage, agnostic to how data is used
- **Model layer**: Statistical inference, agnostic to betting context
- **Betting layer**: EV/edge calculations, consumes model outputs
- **Odds layer**: Market data processing, independent of internal models

This allows swapping the statistical model (e.g., to XGBoost) without touching betting logic, or adding new sportsbooks without modifying the prediction pipeline.

## Roadmap

- [ ] Machine learning ensemble models (XGBoost, gradient boosting)
- [ ] Injury and lineup-based adjustments
- [ ] Historical backtesting framework with walk-forward validation
- [ ] Real-time odds monitoring with alerting
- [ ] Multi-sport expansion (NFL, MLB)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This tool is for educational and research purposes. Sports betting involves financial risk. Please gamble responsibly and in accordance with local laws.
