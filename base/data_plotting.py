import numpy
import pandas as pd
from matplotlib import pyplot as plt

from . import data_processing as dp
import numpy as np
from .utils import logger as logger
import os
import matplotlib.pyplot as plt

logging = logger.get_logger(__name__)

def plot_data_old(
        base_name: str,
        mcd_df: pd.DataFrame,
        abs_df: pd.DataFrame,
        mord_df: pd.DataFrame,
        config: dict,
        output_file_path: str,
        sticks_df: pd.DataFrame = None,) -> None:
    """Plot the MCD, absorption, and MORD data.

    Args:
        base_name (str): Base name for the plots and titles.
        mcd_df (pd.DataFrame): DataFrame containing MCD data.
        abs_df (pd.DataFrame): DataFrame containing absorption data.
        mord_df (pd.DataFrame): DataFrame containing MORD data.
        config (dict): Configuration dictionary with plotting options.
        output_file_path (str): Path to save the output plot.
        sticks_df (pd.DataFrame, optional): DataFrame containing stick spectra. Defaults to None.

    Returns:
        None
    """
    logging.info(f"Starting plot generation for {base_name}")
    # Ensure 'wavelength' column is numeric
    abs_df["wavelength"] = pd.to_numeric(abs_df["wavelength"], errors="coerce")
    mord_df["wavelength"] = pd.to_numeric(mord_df["wavelength"], errors="coerce")

    if sticks_df is not None:
        sticks_df["wavelength"] = pd.to_numeric(
            sticks_df["wavelength"], errors="coerce"
        )

    # Align dataframes to the MCD wavelength range
    wavelength_min = mcd_df["wavelength"].min()
    wavelength_max = mcd_df["wavelength"].max()

    abs_df = abs_df[
        (abs_df["wavelength"] >= wavelength_min)
        & (abs_df["wavelength"] <= wavelength_max)
        ]
    mord_df = mord_df[
        (mord_df["wavelength"] >= wavelength_min)
        & (mord_df["wavelength"] <= wavelength_max)
        ]

    # Clean up MCD data
    mcd_df = mcd_df.dropna(subset=["wavelength", "R_signed_extinction"])
    mcd_df = mcd_df.replace([np.inf, -np.inf], np.nan).dropna()
    if mcd_df.empty:
        logging.error("No valid MCD data available for plotting.")
        return

    # Prepare MCD data for plotting
    mcd_spec = mcd_df["R_signed_extinction"]
    mcd_boxcar = (
        mcd_spec.rolling(window=config["window_size"], center=True)
        .mean()
        .fillna(mcd_spec)
    )
    mcd_X, mcd_Y = dp.interpolate_data(
        mcd_df["wavelength"], mcd_boxcar, config["spline_points"]
    )

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 18))
    fig.subplots_adjust(hspace=0)
    logging.debug("Subplots created.")

    # Clean up absorption data
    abs_df = abs_df.dropna(subset=["wavelength", "intensity_extinction"])
    abs_df = abs_df.replace([np.inf, -np.inf], np.nan).dropna()
    if abs_df.empty:
        logging.error("No valid absorption data available for plotting.")
        return

    # Prepare absorption data for plotting
    abs_spec = abs_df["intensity_extinction"]
    abs_boxcar = (
        abs_spec.rolling(window=config["window_size"], center=True)
        .mean()
        .fillna(abs_spec)
    )
    abs_X, abs_Y = dp.interpolate_data(
        abs_df["wavelength"], abs_boxcar, config["spline_points"]
    )

    # Plot absorption data
    ax1.plot(
        abs_df["wavelength"],
        abs_spec,
        label="Extinction",
        linestyle="-",
        color="blue",
    )
    if config.get("plot_rolling_avg", False):
        ax1.plot(
            abs_df["wavelength"],
            abs_boxcar,
            label="Extinction Boxcar Averaged",
            linestyle="--",
            color="red",
        )
    if config.get("plot_spline", False):
        ax1.plot(
            abs_X,
            abs_Y,
            label="Extinction Spline Interpolation",
            linestyle="-",
            color="green",
        )

    ax1.set_ylabel(r"Molar Extinction, $\epsilon$")
    ax1.grid(False)
    ax1.axhline(y=0, color="dimgrey", linestyle="-")
    ax1.set_title(f"{base_name[:-1]}")

    # Peak picking and plotting (controlled by config flag)
    if config.get("plot_peaks", False):
        peaks, peak_wavelengths, _, peak_intensities = dp.pick_peaks(
            abs_df, column="intensity_extinction", height_percent=1
        )
        ax1.plot(peak_wavelengths, peak_intensities, "ro", label="Peaks")
        for wavelength, intensity in zip(peak_wavelengths, peak_intensities):
            ax1.text(
                wavelength,
                intensity,
                f"{wavelength:.1f}",
                ha="right",
                va="bottom",
                fontsize=10,
                color="red",
            )
        logging.debug("Peaks plotted on absorption graph.")

    ax1.legend()

    # Plot stick spectra if available
    if sticks_df is not None and config.get("plot_sticks", False):
        sticks_df = sticks_df[
            (sticks_df["wavelength"] >= abs_df["wavelength"].min())
            & (sticks_df["wavelength"] <= abs_df["wavelength"].max())
            ]
        max_absorbance = abs_spec.max()
        sticks_df = dp.scale_sticks(sticks_df, max_absorbance)
        ax1_sticks = ax1.twinx()

        for _, row in sticks_df.iterrows():
            ax1_sticks.plot(
                [row["wavelength"], row["wavelength"]],
                [0, row["scaled_strength"]],
                linestyle="-",
                color="black",
            )

        ax1_sticks.set_ylim(ax1.get_ylim())
        ax1_sticks.set_ylabel("Scaled Dipole Strength, Extinction")
        logging.debug("Stick spectra plotted.")

    # Plot MCD data
    if config.get("plot_original", False):
        ax2.plot(
            mcd_df["wavelength"],
            mcd_spec,
            label=r"Measured MCD $\Delta \epsilon / T$",
            linestyle="-",
            color="blue",
        )
    if config.get("plot_rolling_avg", False):
        ax2.plot(
            mcd_df["wavelength"],
            mcd_boxcar,
            label="Rolling Averaged MCD",
            linestyle="--",
            color="red",
        )
    if config.get("plot_spline", False):
        ax2.plot(
            mcd_X,
            mcd_Y,
            label="Spline Interpolation MCD",
            linestyle="-",
            color="green",
        )
    if config.get("plot_error_bars", False):
        ax2.errorbar(
            mcd_df["wavelength"],
            mcd_spec,
            yerr=mcd_df["std_dev_extinction"],
            fmt=",",
            label="Error Bars",
            ecolor="gray",
            alpha=0.5,
        )
    ax2.set_ylabel(r"MCD, $\Delta \epsilon / T$")
    ax2.grid(False)
    ax2.axhline(y=0, color="dimgrey", linestyle="-")
    ax2.legend()

    # Clean up MORD data
    mord_df = mord_df.dropna(subset=["wavelength", "mord"])
    mord_df = mord_df.replace([np.inf, -np.inf], np.nan).dropna()
    if mord_df.empty:
        logging.error("No valid MORD data available for plotting.")
        return

    # Prepare MORD data for plotting
    mord_spec = mord_df["mord"]
    mord_boxcar = (
        mord_spec.rolling(window=config["window_size"], center=True)
        .mean()
        .fillna(mord_spec)
    )
    mord_X, mord_Y = dp.interpolate_data(
        mord_df["wavelength"], mord_boxcar, config["spline_points"]
    )

    # Plot MORD data
    if config.get("plot_original", False):
        ax3.plot(
            mord_df["wavelength"],
            mord_spec,
            label="Measured MORD",
            linestyle="-",
            color="blue",
        )
    if config.get("plot_rolling_avg", False):
        ax3.plot(
            mord_df["wavelength"],
            mord_boxcar,
            label="Rolling Averaged MORD",
            linestyle="--",
            color="red",
        )
    if config.get("plot_spline", False):
        ax3.plot(
            mord_X,
            mord_Y,
            label="Spline Interpolation MORD",
            linestyle="-",
            color="green",
        )
    if config.get("plot_error_bars", False):
        ax3.errorbar(
            mord_df["wavelength"],
            mord_spec,
            yerr=mcd_df["std_dev_extinction"],
            fmt=",",
            label="Error Bars",
            ecolor="gray",
            alpha=0.5,
        )
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel(r"MORD, $\Theta / (M \cdot m \cdot T)$")
    ax3.grid(False)
    ax3.axhline(y=0, color="dimgrey", linestyle="-")
    ax3.legend()

    # Plot derivatives if enabled
    if config.get("plot_derivative", False):
        # First derivative
        abs_df["intensity_derivative"] = np.gradient(
            abs_df["intensity"], abs_df["wavelength"]
        )
        abs_df["intensity_derivative"] *= (
                mcd_spec.max() / abs_df["intensity_derivative"].max()
        )
        ax2.plot(
            abs_df["wavelength"],
            abs_df["intensity_derivative"],
            label="1st Derivative of Abs.",
            linestyle="--",
            color="purple",
        )
        ax2.legend()

        # Second derivative
        abs_df["intensity_2nd_derivative"] = np.gradient(
            abs_df["intensity_derivative"], abs_df["wavelength"]
        )
        abs_df["intensity_2nd_derivative"] *= (
                mord_spec.max() / abs_df["intensity_2nd_derivative"].max()
        )
        ax3.plot(
            abs_df["wavelength"],
            abs_df["intensity_2nd_derivative"],
            label="2nd Derivative of Abs.",
            linestyle="--",
            color="orange",
        )
        ax3.legend()
        logging.debug("Derivatives plotted.")

    # Save and show the plot
    output_plot_path = os.path.join(
        os.path.dirname(output_file_path), f"{base_name}plot.png"
    )
    plt.savefig(output_plot_path)
    logging.info(f"Plot saved to {output_plot_path}")
    plt.show()
    plt.close(fig)
    logging.info("Plot generation completed.")
def plot_gaussian_iterations(x, y, all_fits, num_basis, lowest_bic_idx):
    for fit_idx, fit in enumerate(all_fits):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'b', label='Data with noise', zorder=1)

        # Plot the true original Gaussians - cant in real data
        # for i, g_true in enumerate(gaussians):
        #    plt.plot(x, g_true, 'g-', label=f'True Gaussian {i}', alpha=0.7, zorder=2)

        # Plot the individual fitted Gaussian components for this fit

        #x axis from model to avoid array size errors
        x_fit = fit.userkws['x']
        for i in range(num_basis):
            g_fit = fit.eval_components()[f'g{i}_']
            plt.plot(x_fit, g_fit, label=f'Fitted Gaussian {i} in Fit {fit_idx}', linestyle='--', zorder=4)

        # Plot the composite fit
        plt.plot(x_fit, fit.best_fit, 'r-', label=f'Composite Fit {fit_idx}', zorder=3)
        plt.title(f'Constituent and True Gaussians of N, Iteration: {lowest_bic_idx}, Fit: {fit_idx}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
def plot_true_combined_and_smoothed(x, y, y_smoothed, dd_y_smoothed, dd_y_peaks,  peak_centers, peak_amplitudes, peak_sigmas):
    # Plot the true combined Gaussian curve and smoothed curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='True Combined Gaussian Curve', color='blue', linewidth=2)
    plt.plot(x, y_smoothed, label='Smoothed Combined Curve (y_smoothed)', color='green', linestyle='-', linewidth=2)

    # Plot the numerical derivatives of the smoothed curve
    plt.plot(x, -dd_y_smoothed, label='-1 * Numerical Derivative of y_smoothed', color='green', linestyle='--',
             linewidth=2)
    plt.scatter(x[dd_y_peaks], -dd_y_smoothed[dd_y_peaks], color='red', label='Detected Peaks')

    # make lists for easy removal of elements
    # Plot the initial guesses for each Gaussian curve
    # Add title and labels

    for center, amplitude, sigma in zip(peak_centers, peak_amplitudes, peak_sigmas):
        #filter out by amplitude
        gaussian_guess = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        plt.plot(x, gaussian_guess, label=f'Gaussian guess: Center={center:.2f}, Sigma={sigma:.2f}', linestyle='--')
    plt.title('True Combined Gaussian Curve, Smoothed Curve, and Their Derivatives')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_curve_and_derivative(x, y):
    derivative = np.gradient(y, x)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Combined Gaussian Curve', color='blue')
    plt.plot(x, derivative, label='Numerical Derivative of True Curve', color='orange', linestyle='--')
    plt.title('True Combined Gaussian Curve and its Numerical Derivative')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
def plot_xz_after_gaussian_removal(x,z,reduced_A_result,remaining_derivative_indices,impactful_gaussian_derivative_indices):
    # Plot the fit after the least impactful Gaussians are removed
    plt.figure(figsize=(8, 4))
    plt.plot(x, z, 'b', label='Data', zorder=1)  # smoothed?

    # Plot the individual Gaussian components of the new reduced fit
    for i in remaining_derivative_indices:  # not sure if num basis is going to work here?
        if i in impactful_gaussian_derivative_indices:
            continue
        dg_fit = reduced_A_result.eval_components()[f'g{i}_']
        plt.plot(x, dg_fit, label=f'A-Term {i} after removal', linestyle='--', zorder=3)

    # Plot the composite fit after removal
    plt.plot(x, reduced_A_result.best_fit, 'r-', label='Composite Fit after removal', zorder=2)
    plt.title('Fit after Removing Negligable terms.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
def plot_A_terms(x,z, A_result, remaining_indices):
    plt.figure(figsize=(8, 4))
    plt.plot(x, z, 'b', label='Data with noise', zorder=1)

    # Plot the individual fitted Gaussian derivative components for this fit
    for i in remaining_indices:
        dg_fit = A_result.eval_components()[f'g{i}_']
        # print(A_result.params)
        # Dont display terms with super large sigma
        plt.plot(x, dg_fit, label=f'Fitted Gaussian {i}', linestyle='--', zorder=3)

    # Plot the composite fit
    plt.plot(x, A_result.best_fit, 'r-', label=f'Composite Fit', zorder=2)
    plt.title(f'MCD initial fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
def plot_reduced_result(x, y, num_basis, reduced_result, impactful_gaussian_indices):
    # Plot the fit after the least impactful Gaussians are removed
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b', label='Data with noise', zorder=1)

    # Plot the true original Gaussians again
    # for i, g_true in enumerate(gaussians):
    #    plt.plot(x, g_true, 'g-', label=f'True Gaussian {i}', alpha=0.7, zorder=2)

    # Plot the individual Gaussian components of the new reduced fit
    for i in range(num_basis):
        if i in impactful_gaussian_indices:
            continue
        g_fit = reduced_result.eval_components()[f'g{i}_']
        plt.plot(x, g_fit, label=f'Gaussian {i} after removal', linestyle='--', zorder=4)

    # Plot the composite fit after removal
    plt.plot(x, reduced_result.best_fit, 'r-', label='Composite Fit after removal', zorder=3)
    plt.title('Fit after Removing the Least Impactful Gaussians')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

def plot_bic(avg_bic_values, avg_delta_bic_values):
    # Plot average BIC values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(avg_bic_values) + 1), avg_bic_values, marker='o', color='purple')
    plt.title('Average BIC as a function of number of Gaussians')
    plt.xlabel('Number of basis Gaussians')
    plt.ylabel('Average BIC')
    plt.show()

    # Plot delta BIC values
    plt.figure(figsize=(8, 4))
    if avg_delta_bic_values:
        plt.plot(range(2, len(avg_delta_bic_values) + 2), avg_delta_bic_values, marker='o', color='orange')
        plt.title('Delta BIC as a function of number of Gaussians')
        plt.xlabel('Number of basis Gaussians')
        plt.ylabel('Delta BIC')
        plt.show()
def plot_raw_df(mcd_df):
    plt.plot(mcd_df['wavenumber'], mcd_df['intensity'])
    plt.show()
    plt.plot(mcd_df['wavenumber'], mcd_df['R_signed'])  # To make sure things are scaling.
    plt.show()