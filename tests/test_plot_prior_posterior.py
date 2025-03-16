import numpy as np

from bayesian_calibration.core.plotter import plot_prior_posterior
from bayesian_calibration.core import (
    calibrate_probability,
)
from bayesian_calibration.data import (
    AvailablePriors,
)


def test_plot_prior_posterior():
    prior_params = AvailablePriors.Jeffreys.value

    true_proba = 0.3

    data = [x < true_proba for x in np.random.uniform(size=10)]

    calibration_output = calibrate_probability(data, prior_params)

    # Call the function to ensure it runs without errors
    plot_prior_posterior(calibration_output=calibration_output)
