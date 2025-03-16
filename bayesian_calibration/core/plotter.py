import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from bayesian_calibration.data.calibration_output import CalibrationOutput

sns.set_style(style="whitegrid")


def plot_prior_posterior(
    calibration_output: CalibrationOutput,
):
    """
    Plots the prior and posterior Beta distributions on a given axis.
    """
    x = np.linspace(0, 1, 1000)
    posterior = stats.beta.pdf(
        x,
        calibration_output.posterior_parameters.alpha,
        calibration_output.posterior_parameters.beta,
    )
    prior = stats.beta.pdf(
        x,
        calibration_output.prior_parameters.alpha,
        calibration_output.prior_parameters.beta,
    )

    low_bound = calibration_output.lower_bound
    upper_bound = calibration_output.upper_bound
    confidence_level = calibration_output.confidence_level

    p_mean = calibration_output.mean_probability

    plt.figure(figsize=(6, 6))

    plt.plot(
        x,
        posterior,
        label=f"Posterior distribution $P(A|B)$",
        linewidth=2,
        color="dodgerblue",
    )

    plt.axvline(p_mean, color="salmon", label=f"Mean: {p_mean:.4f}", linestyle="--")
    plt.axvline(
        low_bound,
        color="gray",
        linestyle="--",
        label=f"{confidence_level*100:.0f}% Credible Interval",
        alpha=0.7,
    )
    plt.axvline(upper_bound, color="gray", linestyle="--", alpha=0.7)

    plt.plot(
        x,
        prior,
        color="orangered",
        linestyle="--",
        label="Posterior distribution $P(A)$",
    )

    plt.fill_between(x=x, y1=[0] * len(x), y2=posterior, alpha=0.3, color="dodgerblue")

    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title(
        label=f"Posterior Distribution - Sample Size: {calibration_output.sample_size} - Credible interval size: {upper_bound-low_bound:.2f}",
        fontweight="bold",
    )

    plt.show()
