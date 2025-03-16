from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

from bayesian_calibration.core import (
    calibrate_probability,
)
from bayesian_calibration.data import (
    AvailablePriors,
)
from bayesian_calibration.data import (
    CalibrationOutput,
)


sns.set_style(style="whitegrid")


def generate_binary_data(prob: float, size: int) -> List[bool]:
    """Generates a list of True/False values based on a given probability."""
    return list(np.random.rand(size) < prob)


def plot_prior_posterior(ax, calibration_output: CalibrationOutput):
    """Plots the prior and posterior Beta distributions on the given axis."""
    x = np.linspace(start=0, stop=1, num=1000)
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
    p_mean = calibration_output.mean_probability
    low_bound = calibration_output.lower_bound
    upper_bound = calibration_output.upper_bound
    confidence_level = calibration_output.confidence_level

    ax.plot(x, posterior, label=f"Posterior $P(A|B)$", linewidth=2, color="dodgerblue")
    ax.axvline(p_mean, color="dodgerblue", label=f"Mean: {p_mean:.4f}", linestyle="--")
    ax.axvline(
        low_bound,
        color="gray",
        linestyle="--",
        label=f"{confidence_level*100:.0f}% Credible Interval",
        alpha=0.7,
    )
    ax.axvline(upper_bound, color="gray", linestyle="--", alpha=0.7)
    ax.plot(x, prior, color="orangered", linestyle="--", label="Prior $P(A)$")
    ax.fill_between(x, 0, posterior, alpha=0.3, color="dodgerblue")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.set_title(f"Sample Size: {calibration_output.sample_size}")


def run_experiment(
    prob: float, sizes: List[int], prior: AvailablePriors, save_path: str | None = None
):
    """Runs the calibration and plotting experiment for different sample sizes."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.suptitle(
        t=f"Beta Calibration Experiment (True Probability: {prob}) - Prior: {prior.name}",
        fontsize=14,
        fontweight="bold",
    )

    results = []

    for ax, size in zip(axes.flatten(), sizes):
        data = generate_binary_data(prob, size)
        calibration_output = calibrate_probability(
            data=data, prior_parameters=prior.value
        )
        plot_prior_posterior(ax, calibration_output)

        results.append(
            {
                "Size": size,
                "True proba": prob,
                "EMV": sum(data) / len(data),
                "Calibrated": calibration_output.mean_probability,
            }
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    return pd.DataFrame(data=results)


if __name__ == "__main__":
    # Run the experiment with different sample sizes
    sample_sizes = [2, 3, 5, 10, 25, 50, 100]

    prior = AvailablePriors.Jeffreys

    results = run_experiment(
        prob=0.3,
        sizes=sample_sizes,
        prior=prior,
        save_path="experiments_results/multiple_trials.jpeg",
    )

    results.to_csv("experiments_results/multiple_trials.csv")

    print(results)
