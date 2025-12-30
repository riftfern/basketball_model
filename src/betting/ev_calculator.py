"""Expected value calculation for betting opportunities."""

from dataclasses import dataclass
from typing import Optional

from ..odds.devig import OddsConverter, DevigCalculator
from ..config.settings import settings


@dataclass
class EdgeResult:
    """Result of edge calculation for a betting opportunity."""
    # Probabilities
    model_prob: float           # Our model's probability
    implied_prob: float         # Sportsbook's implied probability (with vig)
    fair_prob: float            # Devigged fair probability

    # Value metrics
    expected_value: float       # EV as decimal (0.05 = 5% EV)
    edge: float                 # Edge over fair prob as decimal

    # Kelly sizing
    kelly_fraction: float       # Full Kelly bet fraction
    kelly_half: float           # Half Kelly
    kelly_quarter: float        # Quarter Kelly

    # Recommendation
    recommended_bet: bool       # Whether to bet based on thresholds
    is_over: bool               # True = over bet, False = under

    # Odds info
    american_odds: int
    decimal_odds: float

    @property
    def ev_pct(self) -> float:
        """Expected value as percentage."""
        return self.expected_value * 100

    @property
    def edge_pct(self) -> float:
        """Edge as percentage."""
        return self.edge * 100

    def kelly_bet(self, bankroll: float, fraction: float = 0.25) -> float:
        """Calculate bet size using Kelly criterion."""
        kelly = self.kelly_fraction * fraction
        kelly = min(kelly, settings.max_bet_pct)  # Cap at max bet
        return bankroll * kelly


class EVCalculator:
    """Calculate expected value and betting edges."""

    def __init__(
        self,
        min_edge: float = None,
        min_ev: float = None,
        devig_method: str = "multiplicative"
    ):
        self.min_edge = min_edge or settings.min_edge
        self.min_ev = min_ev or settings.min_ev
        self.devig_method = devig_method

    def calculate_ev(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calculate expected value.

        EV = P(win) * payout - P(lose) * stake
        EV = P * (decimal_odds - 1) * 1 - (1 - P) * 1
        EV = P * decimal_odds - 1
        """
        return model_prob * decimal_odds - 1

    def kelly_criterion(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calculate Kelly criterion optimal bet fraction.

        f* = (bp - q) / b

        where:
            b = decimal_odds - 1 (net odds)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p

        if b <= 0:
            return 0

        kelly = (b * p - q) / b

        # Don't bet if negative Kelly
        return max(0, kelly)

    def find_edge(
        self,
        model_prob_over: float,
        over_odds: int,
        under_odds: int,
        check_both_sides: bool = True
    ) -> Optional[EdgeResult]:
        """
        Find the best betting edge for a prop.

        Args:
            model_prob_over: Model's probability of over hitting
            over_odds: American odds for over
            under_odds: American odds for under
            check_both_sides: Check both over and under for edges

        Returns:
            EdgeResult for the better side, or None if no edge
        """
        # Calculate fair probabilities
        fair_over, fair_under = DevigCalculator.devig(
            over_odds, under_odds, method=self.devig_method
        )

        results = []

        # Check over
        over_result = self._evaluate_side(
            model_prob=model_prob_over,
            odds=over_odds,
            fair_prob=fair_over,
            is_over=True
        )
        if over_result:
            results.append(over_result)

        # Check under
        if check_both_sides:
            model_prob_under = 1 - model_prob_over
            under_result = self._evaluate_side(
                model_prob=model_prob_under,
                odds=under_odds,
                fair_prob=fair_under,
                is_over=False
            )
            if under_result:
                results.append(under_result)

        if not results:
            return None

        # Return the side with higher EV
        return max(results, key=lambda x: x.expected_value)

    def _evaluate_side(
        self,
        model_prob: float,
        odds: int,
        fair_prob: float,
        is_over: bool
    ) -> Optional[EdgeResult]:
        """Evaluate one side of a prop bet."""
        implied_prob = OddsConverter.american_to_implied(odds)
        decimal_odds = OddsConverter.american_to_decimal(odds)

        # Calculate metrics
        ev = self.calculate_ev(model_prob, decimal_odds)
        edge = model_prob - fair_prob
        kelly = self.kelly_criterion(model_prob, decimal_odds)

        # Check if meets thresholds
        recommended = (
            edge >= self.min_edge and
            ev >= self.min_ev and
            kelly > 0
        )

        return EdgeResult(
            model_prob=model_prob,
            implied_prob=implied_prob,
            fair_prob=fair_prob,
            expected_value=ev,
            edge=edge,
            kelly_fraction=kelly,
            kelly_half=kelly * 0.5,
            kelly_quarter=kelly * 0.25,
            recommended_bet=recommended,
            is_over=is_over,
            american_odds=odds,
            decimal_odds=decimal_odds
        )

    def compare_to_line(
        self,
        model_prob_over: float,
        line: float,
        over_odds: int = -110,
        under_odds: int = -110
    ) -> EdgeResult:
        """
        Compare model probability to a specific line.

        Convenience method when you have standard -110/-110 juice.

        Args:
            model_prob_over: Model's P(stat > line)
            line: The betting line (e.g., 25.5 points)
            over_odds: Over odds (default -110)
            under_odds: Under odds (default -110)
        """
        return self.find_edge(model_prob_over, over_odds, under_odds)
