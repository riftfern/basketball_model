# NBA Player Props Betting Model

A sophisticated statistical modeling platform for identifying profitable NBA player prop betting opportunities. This tool leverages negative binomial distributions, real-time sportsbook odds integration, and contextual performance adjustments to detect positive expected value (+EV) edges in the betting market.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-orange.svg)

## Overview

This project demonstrates applied quantitative analysis for sports betting markets, combining:

- **Statistical Modeling**: Negative binomial distributions with method-of-moments parameter estimation
- **Real-time Data Integration**: Live odds from multiple sportsbooks via The Odds API
- **Contextual Adjustments**: Opponent defense, pace, home/away, rest days, and recency weighting
- **Expected Value Analysis**: Edge detection with Kelly criterion bet sizing
- **Production-Ready CLI**: Full-featured command-line interface with rich terminal output

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
git clone https://github.com/yourusername/basketball_model.git
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

## Roadmap

- [ ] Machine learning ensemble models (XGBoost, neural networks)
- [ ] Injury impact modeling
- [ ] Lineup-based adjustments
- [ ] Historical backtesting framework
- [ ] Real-time odds monitoring and alerting
- [ ] Web dashboard interface

## Skills Demonstrated

This project showcases proficiency in:

- **Quantitative Analysis**: Statistical modeling, probability distributions, expected value theory
- **Software Engineering**: Clean architecture, type safety, database design, API integration
- **Data Engineering**: ETL pipelines, rate limiting, caching strategies
- **Financial Modeling**: Kelly criterion, vig removal, bankroll management
- **Python Ecosystem**: Modern Python practices with type hints, Pydantic, async HTTP

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This tool is for educational and research purposes. Sports betting involves financial risk. Please gamble responsibly and in accordance with local laws.
