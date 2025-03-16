from enum import Enum

from bayesian_calibration.data.beta_prior_parameters import (
    BetaDistributionParameters,
)


class AvailablePriors(Enum):
    Jeffreys = BetaDistributionParameters(0.5, 0.5)
    Uniform = BetaDistributionParameters(1, 1)
    Weak_Gaussian_like = BetaDistributionParameters(2, 2)
    Moderate_Gaussian_like = BetaDistributionParameters(5, 5)
    Strong_Gaussian_like = BetaDistributionParameters(10, 10)
    Very_strong_Gaussian_like = BetaDistributionParameters(20, 20)
    Laplace = BetaDistributionParameters(50, 50)
    Jeffreys_like = BetaDistributionParameters(100, 100)
