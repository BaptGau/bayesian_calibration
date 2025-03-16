from dataclasses import dataclass


@dataclass
class BetaDistributionParameters:
    """
    https://en.wikipedia.org/wiki/Beta_distribution
    """

    alpha: float
    beta: float
