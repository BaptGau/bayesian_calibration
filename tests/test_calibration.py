import pytest
from bayesian_calibration.core import (
    calibrate_probability,
)
from bayesian_calibration.data import (
    AvailablePriors,
)
from bayesian_calibration.data import (
    BetaDistributionParameters,
)
from bayesian_calibration.traits import Probability


class TestCalibrationMethod:
    def setup_method(self):
        self.confidence = Probability(0.95)  # Standard self.confidence interval

    @pytest.mark.parametrize(
        argnames="prior_parameters",
        argvalues=[param.value for param in AvailablePriors],
        ids=[param.name for param in AvailablePriors],
    )
    def test_calibrate_probability_on_small_data(
        self,
        prior_parameters: BetaDistributionParameters,
    ):
        data = [True, False, True, True, False, True]  # 4 successes, 2 failures

        # Call the function to calibrate the probability
        result = calibrate_probability(data, prior_parameters, self.confidence)

        # Assert that the self.confidence interval is returned correctly
        assert (
            result.confidence_level == self.confidence
        ), f"Expected self.confidence level {self.confidence}, got {result.confidence_level}"

        # Assert the mean probability calculation
        expected_mean = result.posterior_parameters.alpha / (
            result.posterior_parameters.alpha + result.posterior_parameters.beta
        )
        assert (
            result.mean_probability == expected_mean
        ), f"Expected mean {expected_mean}, got {result.mean_probability}"

        assert (
            result.lower_bound < result.upper_bound
        ), "Lower bound should be less than upper bound"

        expected_alpha_post = prior_parameters.alpha + 4  # 4 successes
        expected_beta_post = prior_parameters.beta + (6 - 4)  # 2 failures

        assert (
            result.lower_bound <= expected_mean <= result.upper_bound
        ), "Mean should be within the bounds"

        assert (
            result.posterior_parameters.alpha == expected_alpha_post
        ), f"Expected alpha {expected_alpha_post}, got {result.posterior_parameters.alpha}"
        assert (
            result.posterior_parameters.beta == expected_beta_post
        ), f"Expected beta {expected_beta_post}, got {result.posterior_parameters.beta}"

    @pytest.mark.parametrize(
        argnames="prior_parameters",
        argvalues=[param.value for param in AvailablePriors],
        ids=[param.name for param in AvailablePriors],
    )
    def test_calibrate_probability_on_large_data(
        self,
        prior_parameters: BetaDistributionParameters,
    ):
        data_large = [True] * 100 + [False] * 900  # 100 successes, 900 failures

        result_large = calibrate_probability(
            data_large, prior_parameters, self.confidence
        )

        assert 0 <= result_large.lower_bound <= 1, "Lower bound out of range"
        assert 0 <= result_large.upper_bound <= 1, "Upper bound out of range"

        # Check the posterior mean
        expected_mean_large = result_large.posterior_parameters.alpha / (
            result_large.posterior_parameters.alpha
            + result_large.posterior_parameters.beta
        )
        assert (
            result_large.mean_probability == expected_mean_large
        ), f"Expected mean {expected_mean_large}, got {result_large.mean_probability}"

    @pytest.mark.parametrize(
        argnames="prior_parameters",
        argvalues=[param.value for param in AvailablePriors],
        ids=[param.name for param in AvailablePriors],
    )
    def test_calibrate_probability_on_empty_data(
        self,
        prior_parameters: BetaDistributionParameters,
    ):
        data_empty = []  # No data at all
        with pytest.raises(ValueError):
            calibrate_probability(data_empty, prior_parameters, self.confidence)
