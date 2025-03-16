from scipy import stats

from bayesian_calibration.data.beta_prior_parameters import (
    BetaDistributionParameters,
)
from bayesian_calibration.data.calibration_output import (
    CalibrationOutput,
)
from bayesian_calibration.traits import (
    Iterable,
    Probability,
)


def calibrate_probability(
    data: Iterable,
    prior_parameters: BetaDistributionParameters,
    confidence: Probability = 0.95,
) -> CalibrationOutput:
    """
    Computes the Jeffreys interval for a binomial proportion. Calibrate the probability of a binary outcome.

    Parameters
    ----------
    data : Iterable
        A list of binary outcomes (True/False).
    prior_parameters : BetaDistributionParameters
        The prior parameters for the Beta distribution.
    confidence : Probability
        The desired confidence level for the credible interval.


    Returns
    ----------
    ProbabilistOutput: A data object wrapping results.
    """

    if not data:
        raise ValueError("Data shouldn't be empty.")

    successes = sum(data)
    trials = len(data)

    alpha_post = prior_parameters.alpha + successes
    beta_post = prior_parameters.beta + (trials - successes)

    lower_bound = stats.beta.ppf((1 - confidence) / 2, alpha_post, beta_post)
    upper_bound = stats.beta.ppf(1 - (1 - confidence) / 2, alpha_post, beta_post)

    return CalibrationOutput(
        confidence_level=confidence,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prior_parameters=prior_parameters,
        posterior_parameters=BetaDistributionParameters(alpha_post, beta_post),
        sample_size=trials,
    )
