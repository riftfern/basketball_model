"""Negative Binomial distribution model for player props."""

from typing import Optional, Dict, List, Tuple
import numpy as np
from scipy import stats

from ..base import BasePropModel, PropPrediction, PlayerContext
from ...config.constants import PropType, PROP_THRESHOLDS, LEAGUE_AVERAGES


class NegativeBinomialModel(BasePropModel):
    """
    Negative Binomial model for count-based props.

    The Negative Binomial distribution is preferred over Poisson for player props
    because it handles overdispersion (variance > mean), which is common in
    basketball statistics due to varying game scripts, minutes, etc.

    The NB distribution has two parameters:
    - n (or r): number of successes (dispersion parameter)
    - p: probability of success

    Mean = n(1-p)/p
    Variance = n(1-p)/p^2
    """

    def __init__(
        self,
        prop_type: PropType,
        recency_weight: float = 0.15,
        min_games: int = 5
    ):
        self.prop_type = prop_type
        self.recency_weight = recency_weight
        self.min_games = min_games

        # Fitted parameters
        self.n: Optional[float] = None
        self.p: Optional[float] = None
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.n_games: int = 0

    def get_model_name(self) -> str:
        return f"negative_binomial_{self.prop_type.value}"

    def _exponential_weights(self, n: int) -> np.ndarray:
        """Generate exponentially decaying weights for recent games."""
        # Most recent game has highest weight
        weights = np.exp(-self.recency_weight * np.arange(n))
        return weights / weights.sum()

    def _fit_params(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Fit negative binomial parameters using method of moments.

        Returns:
            Tuple of (n, p) parameters
        """
        if weights is None:
            weights = np.ones(len(data)) / len(data)

        # Weighted mean and variance
        mean = np.average(data, weights=weights)
        variance = np.average((data - mean) ** 2, weights=weights)

        # Handle edge case where variance <= mean (use Poisson-like)
        if variance <= mean or mean <= 0:
            # For Poisson-like behavior, use large n
            n = mean * 100
            p = n / (n + mean) if mean > 0 else 0.5
        else:
            # Method of moments for NB
            p = mean / variance
            n = mean * p / (1 - p)

        # Clamp to valid ranges
        n = max(0.1, min(n, 1000))
        p = max(0.001, min(p, 0.999))

        return n, p

    def fit(
        self,
        historical_data: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit model on historical game data.

        Args:
            historical_data: Array of stat values (most recent first)
            weights: Optional custom weights (default: exponential recency)
        """
        if len(historical_data) < self.min_games:
            raise ValueError(
                f"Insufficient data: {len(historical_data)} games < {self.min_games} minimum"
            )

        # Use exponential weights if not provided
        if weights is None:
            weights = self._exponential_weights(len(historical_data))

        self.n, self.p = self._fit_params(historical_data, weights)
        self.mean = self.n * (1 - self.p) / self.p
        self.std = np.sqrt(self.n * (1 - self.p) / (self.p ** 2))
        self.n_games = len(historical_data)

    def _apply_context_adjustments(
        self,
        base_mean: float,
        context: PlayerContext
    ) -> float:
        """
        Adjust predicted mean based on contextual factors.

        Key adjustments:
        1. Opponent defensive rating (vs league average)
        2. Pace adjustment
        3. Home/away factor
        4. Rest days / back-to-back
        """
        adjusted = base_mean

        # 1. Opponent defense adjustment
        if context.opponent_def_rating:
            league_avg_def = LEAGUE_AVERAGES["def_rating"]
            def_factor = context.opponent_def_rating / league_avg_def

            # Different prop types affected differently by defense
            if self.prop_type in [PropType.POINTS, PropType.THREES]:
                # Scoring props more affected by defense
                adjusted *= def_factor
            elif self.prop_type == PropType.REBOUNDS:
                # Rebounds moderately affected
                adjusted *= (1 + (def_factor - 1) * 0.3)
            elif self.prop_type == PropType.ASSISTS:
                # Assists slightly affected
                adjusted *= (1 + (def_factor - 1) * 0.2)
            # Steals, blocks, turnovers - minimal adjustment

        # 2. Pace adjustment
        if context.opponent_pace:
            league_avg_pace = LEAGUE_AVERAGES["pace"]
            pace_factor = context.opponent_pace / league_avg_pace
            # Square root for moderate effect
            adjusted *= pace_factor ** 0.5

        # 3. Home/away (~2% home boost)
        if context.is_home:
            adjusted *= 1.02

        # 4. Rest days
        if context.is_back_to_back:
            # ~3-5% decrease on back-to-backs
            adjusted *= 0.97
        elif context.rest_days and context.rest_days >= 3:
            # Slight boost with extended rest
            adjusted *= 1.01

        return adjusted

    def predict(self, context: Optional[PlayerContext] = None) -> PropPrediction:
        """
        Generate prediction with full probability distribution.

        Args:
            context: Optional game context for adjustments

        Returns:
            PropPrediction with mean, distribution, and threshold probabilities
        """
        if self.mean is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Apply contextual adjustments if provided
        if context:
            adjusted_mean = self._apply_context_adjustments(self.mean, context)
        else:
            adjusted_mean = self.mean

        # Recalculate NB parameters with adjusted mean
        # Keep original dispersion (variance/mean ratio)
        dispersion = self.n if self.n else 10
        variance = adjusted_mean + (adjusted_mean ** 2) / dispersion

        p = adjusted_mean / variance if variance > 0 else 0.5
        n = adjusted_mean * p / (1 - p) if (1 - p) > 0 else dispersion

        # Clamp parameters
        n = max(0.1, min(n, 1000))
        p = max(0.001, min(p, 0.999))

        # Pre-compute threshold probabilities
        thresholds = PROP_THRESHOLDS.get(self.prop_type, [])
        threshold_probs = {}

        for thresh in thresholds:
            # P(X > threshold) = 1 - P(X <= threshold) = 1 - CDF(threshold)
            prob_over = 1 - stats.nbinom.cdf(thresh, n, p)
            threshold_probs[thresh] = prob_over

        return PropPrediction(
            prop_type=self.prop_type.value,
            predicted_mean=adjusted_mean,
            predicted_std=np.sqrt(variance),
            distribution_type="negative_binomial",
            distribution_params={"n": n, "p": p},
            threshold_probs=threshold_probs
        )

    def simulate(self, n_simulations: int = 10000, context: Optional[PlayerContext] = None) -> np.ndarray:
        """
        Run Monte Carlo simulations.

        Args:
            n_simulations: Number of simulations to run
            context: Optional game context

        Returns:
            Array of simulated values
        """
        pred = self.predict(context)
        n = pred.distribution_params["n"]
        p = pred.distribution_params["p"]

        return np.random.negative_binomial(n, p, size=n_simulations)


def create_prop_models() -> Dict[PropType, NegativeBinomialModel]:
    """Create models for all prop types."""
    return {
        prop_type: NegativeBinomialModel(prop_type)
        for prop_type in [
            PropType.POINTS,
            PropType.REBOUNDS,
            PropType.ASSISTS,
            PropType.STEALS,
            PropType.BLOCKS,
            PropType.THREES,
            PropType.TURNOVERS
        ]
    }
