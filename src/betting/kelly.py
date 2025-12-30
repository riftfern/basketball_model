"""Kelly criterion and bankroll management."""

from dataclasses import dataclass
from typing import List, Optional
from ..config.settings import settings


@dataclass
class BetSizing:
    """Recommended bet sizing."""
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    recommended: float
    bet_amount: float
    bankroll: float

    @property
    def risk_pct(self) -> float:
        """Percentage of bankroll at risk."""
        return (self.bet_amount / self.bankroll) * 100


class KellyCalculator:
    """
    Kelly criterion calculator for optimal bet sizing.

    The Kelly criterion maximizes long-term growth rate but
    can be volatile. Using fractional Kelly (25-50%) reduces
    variance while still capturing most of the edge.
    """

    def __init__(
        self,
        default_fraction: float = None,
        max_bet_pct: float = None
    ):
        self.default_fraction = default_fraction or settings.kelly_fraction
        self.max_bet_pct = max_bet_pct or settings.max_bet_pct

    def calculate(
        self,
        prob: float,
        decimal_odds: float,
        bankroll: float,
        kelly_fraction: Optional[float] = None
    ) -> BetSizing:
        """
        Calculate bet sizing using Kelly criterion.

        Args:
            prob: Probability of winning
            decimal_odds: Decimal odds (e.g., 1.91 for -110)
            bankroll: Current bankroll
            kelly_fraction: Fraction of Kelly to use (default from settings)

        Returns:
            BetSizing with recommended amounts
        """
        fraction = kelly_fraction or self.default_fraction

        # Full Kelly
        b = decimal_odds - 1
        p = prob
        q = 1 - p

        full = (b * p - q) / b if b > 0 else 0
        full = max(0, full)  # Don't go negative

        # Fractional Kelly
        half = full * 0.5
        quarter = full * 0.25
        recommended = full * fraction

        # Apply max bet cap
        capped = min(recommended, self.max_bet_pct)
        bet_amount = bankroll * capped

        return BetSizing(
            full_kelly=full,
            half_kelly=half,
            quarter_kelly=quarter,
            recommended=capped,
            bet_amount=bet_amount,
            bankroll=bankroll
        )

    def optimal_fraction_for_variance(
        self,
        target_drawdown: float = 0.20,
        edge: float = 0.05,
        num_bets: int = 100
    ) -> float:
        """
        Calculate Kelly fraction to limit expected drawdown.

        Args:
            target_drawdown: Maximum acceptable drawdown (e.g., 0.20 = 20%)
            edge: Average edge per bet
            num_bets: Expected number of bets

        Returns:
            Recommended Kelly fraction
        """
        import numpy as np

        # Rough approximation: drawdown ~ sqrt(n) * kelly_fraction * volatility
        # This is simplified; real calculation would need simulation
        vol_factor = np.sqrt(num_bets) * 0.5  # Approximate volatility factor

        if vol_factor == 0:
            return self.default_fraction

        suggested = target_drawdown / vol_factor

        # Clamp to reasonable range
        return max(0.1, min(0.5, suggested))

    def simulate_growth(
        self,
        prob: float,
        decimal_odds: float,
        kelly_fraction: float,
        n_bets: int = 1000,
        n_simulations: int = 1000
    ) -> dict:
        """
        Simulate bankroll growth with given parameters.

        Returns:
            Dictionary with growth statistics
        """
        import numpy as np

        results = []
        bet_size = kelly_fraction

        for _ in range(n_simulations):
            bankroll = 1.0  # Start with $1 for percentage growth

            for _ in range(n_bets):
                # Win or lose
                if np.random.random() < prob:
                    bankroll *= (1 + bet_size * (decimal_odds - 1))
                else:
                    bankroll *= (1 - bet_size)

            results.append(bankroll)

        results = np.array(results)

        return {
            "median_growth": float(np.median(results)),
            "mean_growth": float(np.mean(results)),
            "std_growth": float(np.std(results)),
            "min_growth": float(np.min(results)),
            "max_growth": float(np.max(results)),
            "prob_profit": float((results > 1).mean()),
            "prob_double": float((results > 2).mean()),
            "prob_ruin": float((results < 0.1).mean())  # Lost 90%+
        }
