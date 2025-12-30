"""The Odds API integration for fetching sportsbook lines."""

import httpx
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..config.settings import settings


@dataclass
class PropLine:
    """A player prop line from a sportsbook."""
    player_name: str
    prop_type: str  # "player_points", "player_rebounds", etc.
    line: float
    over_odds: int  # American odds
    under_odds: int
    sportsbook: str
    game_id: str
    home_team: str
    away_team: str
    commence_time: datetime


class OddsAPIClient:
    """
    Client for The Odds API.

    Docs: https://the-odds-api.com/liveapi/guides/v4/
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    # Map Odds API prop types to our internal types
    PROP_TYPE_MAP = {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "threes",
        "player_steals": "steals",
        "player_blocks": "blocks",
        "player_turnovers": "turnovers",
        "player_points_rebounds_assists": "pra",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            raise ValueError("Odds API key required. Set NBA_ODDS_API_KEY env var.")

        self.client = httpx.Client(timeout=30.0)
        self._remaining_requests = None
        self._used_requests = None

    @property
    def requests_remaining(self) -> Optional[int]:
        """Number of API requests remaining this month."""
        return self._remaining_requests

    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """Make API request and track usage."""
        params = params or {}
        params["apiKey"] = self.api_key

        response = self.client.get(f"{self.BASE_URL}{endpoint}", params=params)

        # Track API usage from headers
        self._remaining_requests = int(response.headers.get("x-requests-remaining", 0))
        self._used_requests = int(response.headers.get("x-requests-used", 0))

        response.raise_for_status()
        return response.json()

    def get_nba_games(self) -> List[Dict]:
        """Get upcoming NBA games."""
        return self._make_request("/sports/basketball_nba/odds", {
            "regions": "us",
            "markets": "h2h",
        })

    def get_player_props(
        self,
        event_id: str,
        markets: List[str] = None
    ) -> Dict:
        """
        Get player props for a specific game.

        Args:
            event_id: The game/event ID from get_nba_games()
            markets: List of prop markets. Default: points, rebounds, assists
        """
        if markets is None:
            markets = [
                "player_points",
                "player_rebounds",
                "player_assists",
                "player_threes"
            ]

        return self._make_request(
            f"/sports/basketball_nba/events/{event_id}/odds",
            {
                "regions": "us",
                "markets": ",".join(markets),
                "oddsFormat": "american"
            }
        )

    def get_all_player_props(
        self,
        prop_types: List[str] = None
    ) -> List[PropLine]:
        """
        Get all player props for today's NBA games.

        Returns:
            List of PropLine objects with odds from all sportsbooks
        """
        if prop_types is None:
            prop_types = ["player_points", "player_rebounds", "player_assists"]

        # First get all games
        games = self.get_nba_games()

        all_props = []

        for game in games:
            event_id = game["id"]
            home_team = game["home_team"]
            away_team = game["away_team"]
            commence_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))

            try:
                props_data = self.get_player_props(event_id, prop_types)
            except Exception as e:
                print(f"Error fetching props for {away_team} @ {home_team}: {e}")
                continue

            # Parse bookmaker data
            for bookmaker in props_data.get("bookmakers", []):
                sportsbook = bookmaker["key"]

                for market in bookmaker.get("markets", []):
                    prop_type = market["key"]

                    # Group outcomes by player
                    player_outcomes = {}
                    for outcome in market.get("outcomes", []):
                        player = outcome.get("description", "")
                        if not player:
                            continue

                        if player not in player_outcomes:
                            player_outcomes[player] = {"line": outcome.get("point")}

                        if outcome["name"] == "Over":
                            player_outcomes[player]["over"] = outcome["price"]
                        elif outcome["name"] == "Under":
                            player_outcomes[player]["under"] = outcome["price"]

                    # Create PropLine objects
                    for player, data in player_outcomes.items():
                        if "over" in data and "under" in data and data.get("line"):
                            all_props.append(PropLine(
                                player_name=player,
                                prop_type=self.PROP_TYPE_MAP.get(prop_type, prop_type),
                                line=float(data["line"]),
                                over_odds=int(data["over"]),
                                under_odds=int(data["under"]),
                                sportsbook=sportsbook,
                                game_id=event_id,
                                home_team=home_team,
                                away_team=away_team,
                                commence_time=commence_time
                            ))

        return all_props

    def get_best_lines(self, props: List[PropLine]) -> Dict[str, Dict[float, PropLine]]:
        """
        Find the best available line for each player/prop/line combo.

        Groups by player+prop+line and returns the best over and under odds.
        """
        best = {}

        for prop in props:
            key = f"{prop.player_name}|{prop.prop_type}"

            if key not in best:
                best[key] = {}

            line = prop.line
            if line not in best[key]:
                best[key][line] = {
                    "best_over": prop,
                    "best_under": prop,
                    "all_books": [prop]
                }
            else:
                best[key][line]["all_books"].append(prop)

                # Update best over
                if prop.over_odds > best[key][line]["best_over"].over_odds:
                    best[key][line]["best_over"] = prop

                # Update best under
                if prop.under_odds > best[key][line]["best_under"].under_odds:
                    best[key][line]["best_under"] = prop

        return best
