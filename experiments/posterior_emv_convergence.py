import numpy as np

from bayesian_calibration.core import (
    calibrate_probability,
)
from bayesian_calibration.data import AvailablePriors

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style(style="whitegrid")


def generate_binary_data(prob: float, size: int) -> list[bool]:
    """Generates a list of True/False values based on a given probability."""
    return list(np.random.rand(size) < prob)


def plot_experiment_results(results: list[float], save_path: str | None = None):
    """Plots the results of the experiment."""
    plt.figure(figsize=(10, 6))
    plt.plot(results, marker="o", linestyle="-", color="chartreuse")
    plt.fill_between(
        range(len(results)), [0] * len(results), results, color="chartreuse", alpha=0.3
    )
    plt.xlabel("Sample Size")
    plt.ylabel("EMV - Posterior Mean")
    plt.title("Convergence of the Posterior Mean to the EMV")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def run_experiment(
    prob: float, prior: AvailablePriors, save_path: str | None = None
) -> list[float]:
    """Runs the calibration and plotting experiment for different sample sizes."""

    results = []

    sizes = np.arange(1, 51, 1)

    for size in sizes:
        data = generate_binary_data(prob, size)
        calibration_output = calibrate_probability(
            data=data, prior_parameters=prior.value
        )

        emv = sum(data) / len(data)
        results.append(np.abs(emv - calibration_output.mean_probability))

    plot_experiment_results(results=results, save_path=save_path)

    return results


if __name__ == "__main__":
    run_experiment(
        prob=0.3,
        prior=AvailablePriors.Jeffreys,
        save_path="experiments_results/emv_convergence.jpeg",
    )
