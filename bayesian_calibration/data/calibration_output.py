from dataclasses import dataclass, field

from bayesian_calibration.data.beta_prior_parameters import (
    BetaDistributionParameters,
)
from bayesian_calibration.traits import Probability


@dataclass
class CalibrationOutput:
    confidence_level: Probability
    lower_bound: Probability
    upper_bound: Probability
    prior_parameters: BetaDistributionParameters
    posterior_parameters: BetaDistributionParameters
    sample_size: int
    mean_probability: Probability = field(init=False)
    median_probability: Probability = field(init=False)

    def __post_init__(self):
        self.mean_probability = Probability(
            self.posterior_parameters.alpha
            / (self.posterior_parameters.alpha + self.posterior_parameters.beta)
        )
        self.median_probability = Probability(
            (self.posterior_parameters.alpha - 1 / 3)
            / (self.posterior_parameters.alpha + self.posterior_parameters.beta - 2 / 3)
        )
