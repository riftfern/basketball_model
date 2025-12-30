"""Base classes for prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
from scipy import stats


@dataclass
class PlayerContext:
    """Contextual factors for a game prediction."""
    player_id: int
    opponent_team_id: Optional[int] = None
    is_home: bool = True
    rest_days: int = 1
    is_back_to_back: bool = False

    # Opponent stats (filled in during prediction)
    opponent_def_rating: Optional[float] = None
    opponent_pace: Optional[float] = None

    # Player recent form
    minutes_avg: Optional[float] = None


@dataclass
class PropPrediction:
    """Model prediction output with full distribution."""
    prop_type: str
    predicted_mean: float
    predicted_std: float
    distribution_type: str
    distribution_params: Dict

    # Pre-computed threshold probabilities
    threshold_probs: Dict[float, float] = field(default_factory=dict)

    def prob_over(self, threshold: float) -> float:
        """Calculate P(X > threshold)."""
        # Check pre-computed first
        if threshold in self.threshold_probs:
            return self.threshold_probs[threshold]

        # Calculate based on distribution
        if self.distribution_type == "poisson":
            lam = self.distribution_params["lambda"]
            return 1 - stats.poisson.cdf(threshold, lam)

        elif self.distribution_type == "negative_binomial":
            n = self.distribution_params["n"]
            p = self.distribution_params["p"]
            return 1 - stats.nbinom.cdf(threshold, n, p)

        elif self.distribution_type == "normal":
            mu = self.distribution_params["mu"]
            sigma = self.distribution_params["sigma"]
            return 1 - stats.norm.cdf(threshold, mu, sigma)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution_type}")

    def prob_under(self, threshold: float) -> float:
        """Calculate P(X < threshold)."""
        return 1 - self.prob_over(threshold)

    def get_percentile(self, percentile: float) -> float:
        """Get the value at a given percentile (0-100)."""
        q = percentile / 100

        if self.distribution_type == "poisson":
            return stats.poisson.ppf(q, self.distribution_params["lambda"])

        elif self.distribution_type == "negative_binomial":
            n = self.distribution_params["n"]
            p = self.distribution_params["p"]
            return stats.nbinom.ppf(q, n, p)

        elif self.distribution_type == "normal":
            mu = self.distribution_params["mu"]
            sigma = self.distribution_params["sigma"]
            return stats.norm.ppf(q, mu, sigma)

        return self.predicted_mean

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "prop_type": self.prop_type,
            "predicted_value": self.predicted_mean,
            "predicted_std": self.predicted_std,
            "distribution_type": self.distribution_type,
            "distribution_params": self.distribution_params,
            "threshold_probs": self.threshold_probs
        }


class BasePropModel(ABC):
    """Abstract base class for prop prediction models."""

    @abstractmethod
    def fit(self, historical_data: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """Fit model on historical data."""
        pass

    @abstractmethod
    def predict(self, context: Optional[PlayerContext] = None) -> PropPrediction:
        """Generate prediction with full distribution."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier."""
        pass
