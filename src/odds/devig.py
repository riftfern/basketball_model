"""Odds conversion and devigging utilities."""

from typing import Tuple
import numpy as np


class OddsConverter:
    """Convert between different odds formats."""

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))

    @staticmethod
    def decimal_to_american(decimal: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal >= 2.0:
            return int(round((decimal - 1) * 100))
        else:
            return int(round(-100 / (decimal - 1)))

    @staticmethod
    def american_to_implied(american: int) -> float:
        """Convert American odds to implied probability."""
        if american > 0:
            return 100 / (american + 100)
        else:
            return abs(american) / (abs(american) + 100)

    @staticmethod
    def implied_to_american(prob: float) -> int:
        """Convert implied probability to American odds."""
        if prob <= 0 or prob >= 1:
            raise ValueError("Probability must be between 0 and 1")

        if prob >= 0.5:
            return int(round(-100 * prob / (1 - prob)))
        else:
            return int(round(100 * (1 - prob) / prob))

    @staticmethod
    def decimal_to_implied(decimal: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1 / decimal


class DevigCalculator:
    """
    Remove vig (juice) from sportsbook odds to find fair probabilities.

    Different methods have different assumptions:
    - Multiplicative: Assumes vig is applied equally
    - Power: Better for favorites/underdogs
    - Shin: Accounts for insider information
    """

    @staticmethod
    def multiplicative_devig(over_odds: int, under_odds: int) -> Tuple[float, float]:
        """
        Multiplicative method: Scale probabilities proportionally.

        Most common method, assumes equal vig on both sides.

        Args:
            over_odds: American odds for over
            under_odds: American odds for under

        Returns:
            Tuple of (fair_over_prob, fair_under_prob)
        """
        over_implied = OddsConverter.american_to_implied(over_odds)
        under_implied = OddsConverter.american_to_implied(under_odds)

        total = over_implied + under_implied  # > 1 due to vig

        fair_over = over_implied / total
        fair_under = under_implied / total

        return fair_over, fair_under

    @staticmethod
    def power_devig(over_odds: int, under_odds: int) -> Tuple[float, float]:
        """
        Power method: More accurate for favorites/underdogs.

        Raises implied probabilities to a power to normalize.
        Better at handling asymmetric vig.
        """
        from scipy.optimize import brentq

        over_implied = OddsConverter.american_to_implied(over_odds)
        under_implied = OddsConverter.american_to_implied(under_odds)

        def objective(k):
            return over_implied ** k + under_implied ** k - 1

        try:
            k = brentq(objective, 0.5, 2.0)
            fair_over = over_implied ** k
            fair_under = under_implied ** k
        except ValueError:
            # Fallback to multiplicative
            return DevigCalculator.multiplicative_devig(over_odds, under_odds)

        return fair_over, fair_under

    @staticmethod
    def worst_case_devig(over_odds: int, under_odds: int) -> Tuple[float, float]:
        """
        Worst-case method: Assumes all vig is on your side.

        Most conservative for finding edges.
        """
        over_implied = OddsConverter.american_to_implied(over_odds)
        under_implied = OddsConverter.american_to_implied(under_odds)

        total = over_implied + under_implied
        vig = total - 1

        # Assume all vig is against the side we're betting
        fair_over = over_implied - vig
        fair_under = under_implied - vig

        # Clamp to valid range
        fair_over = max(0.01, min(0.99, fair_over))
        fair_under = max(0.01, min(0.99, fair_under))

        return fair_over, fair_under

    @classmethod
    def devig(
        cls,
        over_odds: int,
        under_odds: int,
        method: str = "multiplicative"
    ) -> Tuple[float, float]:
        """
        Devig using specified method.

        Args:
            over_odds: American odds for over
            under_odds: American odds for under
            method: "multiplicative", "power", or "worst_case"

        Returns:
            Tuple of (fair_over_prob, fair_under_prob)
        """
        methods = {
            "multiplicative": cls.multiplicative_devig,
            "power": cls.power_devig,
            "worst_case": cls.worst_case_devig
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Use: {list(methods.keys())}")

        return methods[method](over_odds, under_odds)

    @staticmethod
    def calculate_vig(over_odds: int, under_odds: int) -> float:
        """
        Calculate the total vig/juice in the market.

        Returns:
            Vig as a percentage (e.g., 4.5 for 4.5% vig)
        """
        over_implied = OddsConverter.american_to_implied(over_odds)
        under_implied = OddsConverter.american_to_implied(under_odds)
        return (over_implied + under_implied - 1) * 100
