import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
import lmfit
import time
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
from lmfit import Model
import os

# Define all the constants

# Test data generation.
NUM_GAUSSIANS = 3  # Number of Gaussians to generate
NOISE_LEVEL = 0.01  # Noise level to add to the generated Gaussian curve
NUM_X_VALUES = 500  # Number of points in the x-axis array
X_MIN = 0  # Minimum value for the x range
X_MAX = 60  # Maximum value for the x range
SEED = 479806  # seed for reproducibility.
MIN_DISTANCE = 5  # simulate the 'bandwidth' of the fake instrument.

# Smoothing
WINDOW_LENGTH = 5  # Window length for Savitzky-Golay smoothing (datapoints?) (relate to bandwidth?)
POLYORDER = 4  # Polynomial order for Savitzky-Golay smoothing

# Peak picking
HEIGHT_THRESHOLD = 0.04  # Minimum height threshold for peak detection - should be greater than noise after smoothing. Trouble is that the negative side bands likely mean zero isn't at zero.
DISTANCE = 5  # The minimum distance, in number of samples, between peaks. - This should be related to bandwidth for certain
PROMINENCE_PERECENT = 0.04  # Prominence is here as a multiple of max height. What is (topographic) prominence? It is "the minimum height necessary to descend to get from the summit to any higher terrain", as it can be seen here

# Fitting
MAX_BASIS_GAUSSIANS = 10  # Maximum number of basis Gaussians for fitting
NUM_GUESSES = 1  # Number of guesses for each basis fitting - created this with the idea that I would want to allow for some random walk over guess space ? For now keep to 1.
VARY_CENTERS = False
DELTA_BIC_THRESHOLD = 0  # Threshold for detecting when BIC levels out - depends on the number of points. Need to normalize somehow. Set to zero to effectively bypass.
THRESHOLD_PERCENT = 0.1  # Threshold under which a basis curve is not contributing to the fit, and therefore is removed. This should probably be maximally 100/N, N is the number of (reasonable) basis curves. In experience, the ones that don't contribute tend to be very small say less than 1%
PERCENTAGE_RANGE = 1  # The percentage by which the initial parameters will be allowed to relax on re-fitting after removing poor curves.
PERCENT_RANGE_X = 1
TOLERANCE_X = 2  # how close centers can be as a percent of the overall x values and be considered the same.

# Constants
SMALL_FWHM_FACTOR = 2.355  # Conversion factor from FWHM to sigma
tiny = 1.0e-15
log2 = np.log(2)
s2pi = np.sqrt(2 * np.pi)
s2 = np.sqrt(2.0)


# Define a function to group close centers, using tolerance as a percentage of the x range
# Define a function to group close centers, using tolerance as a percentage of the x range
def group_centers(dipole_params, aterm_params, x_values, tolerance_percentage=5):
    """Groups transitions together. Right now, also converts a terms to be positive or negative according to convention. """
    grouped_params = []

    # Calculate the total x-value range (wavenumber range)
    x_range = max(x_values) - min(x_values)
    tolerance = (tolerance_percentage / 100) * x_range  # tolerance as a percentage of the x range

    # Create lists to store results
    centers_dipole = [(name, param.value) for name, param in dipole_params.items() if 'center' in name]
    centers_aterm = [(name, param.value) for name, param in aterm_params.items() if 'center' in name]

    # Pair the closest centers from dipole and aterm, within the calculated tolerance
    for d_center_name, d_center_value in centers_dipole:
        for a_center_name, a_center_value in centers_aterm:
            if abs(d_center_value - a_center_value) <= tolerance:
                # Get the corresponding amplitudes
                d_amplitude = dipole_params[d_center_name.replace('center', 'amplitude')].value
                a_amplitude = aterm_params[a_center_name.replace('center', 'amplitude')].value

                # Store the values in the list
                grouped_params.append({
                    'Electronic Dipole Center (D_0(x))': d_center_value,
                    'A-Term Center (A_1(x))': a_center_value,
                    'Amplitude (D_0)': d_amplitude,
                    'Amplitude (A_1)': - a_amplitude,
                    'A/D Ratio': - a_amplitude / d_amplitude if d_amplitude != 0 else None
                })

    return grouped_params


def custom_gaussian_old(x, amplitude, center, sigma):
    """Gaussian function."""
    return (amplitude / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def custom_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude / (max(tiny, s2pi * sigma)))
            * np.exp(-(1.0 * x - center) ** 2 / max(tiny, (
                        2 * sigma ** 2))))  # here, why do they multiply by 1.0? Is it to set it to a float always?


# Function to generate a single Gaussian
def gaussian_old(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))


def gaussian(x, mu, gamma, a_0):
    "Returns a normalized gaussian function. Normalized to 1"
    return a_0 * ((2 * np.sqrt(np.log(2))) / (gamma * np.sqrt(np.pi))) * np.exp(
        -((4 * np.log(2)) / (gamma ** 2)) * (x - mu) ** 2)


def stable_gaussian(x, mu, gamma, a_0):
    """Returns a numerically stable gaussian.
    In the case that non-constant values are found (i.e. not Tiny), then should be normalized to 1.
    Some choices are not clear - like multplying by 4.0 instead of 4.
    We choose not to provide default values because we wish to fail on not passing good values."""
    return a_0 * ((2 * np.sqrt(np.log(2))) / max(tiny, (gamma * np.sqrt(np.pi)))) * np.exp(
        -((4.0 * max(tiny, (np.log(2)) / (gamma ** 2)))) * (
                    x - mu) ** 2)  # here we multiply by 4.0 for the same reason that they multiply by 1.0. Maybe this sets as a float.


def stable_gaussian_sigma(x, amplitude, center, sigma):
    """Returns a numerically stable gaussian. Takes arguments in terms of sigma.
    In the case that non-constant values are found (i.e. not Tiny), then should be normalized to 1.
    Some choices are not clear - like multplying by 4.0 instead of 4.
    We choose not to provide default values because we wish to fail on not passing good values."""
    gamma = sigma * 2 * np.sqrt(2 * np.log(2))
    a_0 = amplitude
    mu = center
    return a_0 * ((2 * np.sqrt(np.log(2))) / max(tiny, (gamma * np.sqrt(np.pi)))) * np.exp(
        -((4.0 * max(tiny, (np.log(2)) / (gamma ** 2)))) * (
                    x - mu) ** 2)  # here we multiply by 4.0 for the same reason that they multiply by 1.0. Maybe this sets as a float.


def gaussian_derivative(x, mu, gamma, a_0):
    "Returns the normalized first derivative of a gaussian function. Normalized such that the absolute value of the area under the curve is 1"
    return ((- a_0 * 4 * np.log(2) * (x - mu)) / (gamma ** 2)) * np.exp(
        -((4 * np.log(2)) / (gamma ** 2)) * (x - mu) ** 2)
    # The following is the not normalized form. Instead, normalize by multiplying by x?.
    # return ((a_0 * -16 * np.log(2) * np.sqrt(np.log(2)) * (x - mu))/(gamma**3 * np.sqrt(np.pi))) * np.exp(-((4 * np.log(2))/(gamma**2)) * (x - mu)**2 )


def stable_gaussian_derivative_sigma(x, amplitude, center, sigma):
    "Returns the normalized first derivative of a gaussian function. Normalized such that f(x) * x = -1"
    gamma = sigma * 2 * np.sqrt(2 * np.log(2))
    a_0 = amplitude
    mu = center
    # return ((- a_0 * 4 * np.log(2) * (x - mu))/ max(tiny, gamma**2)) * np.exp(-((4.0 * max(tiny, (np.log(2))/(gamma**2)))) * (x - mu)**2)
    return (a_0 * -16.0 * np.log(2) * np.sqrt(np.log(2)) * (x - mu)) / (
        max(tiny, (gamma ** 3 * np.sqrt(np.pi)))) * np.exp(-((4.0 * max(tiny, (np.log(2)) / (gamma ** 2)))) * (
                x - mu) ** 2)  # this function is not normalized to be area under curve of 1.


class CustomGaussianModel_default(Model):
    """A custom Gaussian model replicating the default lmfit GaussianModel."""

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    height_factor = 1 / np.sqrt(2 * np.pi)

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        """
        Initialize the CustomGaussianModel.

        Parameters:
        - independent_vars (list): List of independent variable names.
        - prefix (str): Prefix for parameter names.
        - nan_policy (str): Handling of NaN values.
        """
        super().__init__(custom_gaussian, prefix=prefix, independent_vars=independent_vars, nan_policy=nan_policy,
                         **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """Set parameter hints for amplitude, center, sigma, fwhm, and height."""
        self.set_param_hint('amplitude', min=0)  # Amplitude must be positive
        self.set_param_hint('center')
        self.set_param_hint('sigma', min=0)  # Sigma must be positive

        # Expressions for derived parameters fwhm and height
        self.set_param_hint('fwhm', expr=f'{self.prefix}sigma * {self.fwhm_factor}')
        # self.set_param_hint('height', expr=f'{self.prefix}amplitude / ({self.prefix}sigma * {self.height_factor})') double check.

    def guess(self, data, x, negative=False, **kwargs):
        """
        Estimate initial parameter values from data.

        Parameters:
        - data (array): The dependent data.
        - x (array): The independent variable.
        - negative (bool): Whether to invert the data for peak finding.

        Returns:
        - Parameters: An lmfit.Parameters object with initial guesses.
        """
        # Guess the peak amplitude
        amplitude_guess = np.max(data) if not negative else np.min(data)
        # Guess the peak center
        center_guess = x[np.argmax(data)] if not negative else x[np.argmin(data)]
        # Guess sigma as a fraction of the total range
        sigma_guess = (x.max() - x.min()) / 6.0  # Rough estimate for sigma

        params = self.make_params(amplitude=amplitude_guess, center=center_guess, sigma=sigma_guess)
        return params


class CustomGaussianModel(Model):
    """A custom Gaussian model replicating the default lmfit GaussianModel."""

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    height_factor = 1 / np.sqrt(2 * np.pi)

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        """
        Initialize the CustomGaussianModel.

        Parameters:
        - independent_vars (list): List of independent variable names.
        - prefix (str): Prefix for parameter names.
        - nan_policy (str): Handling of NaN values.
        """
        super().__init__(stable_gaussian_sigma, prefix=prefix, independent_vars=independent_vars, nan_policy=nan_policy,
                         **kwargs)
        # it looks like we have a nan issue. - solved by matching param names. duh.
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """Set parameter hints for amplitude, center, sigma, fwhm, and height."""
        self.set_param_hint('amplitude', min=0)  # Amplitude must be positive
        self.set_param_hint('center')
        self.set_param_hint('sigma', min=0)  # Sigma must be positive

        # Expressions for derived parameters fwhm and height
        self.set_param_hint('fwhm', expr=f'{self.prefix}sigma * {self.fwhm_factor}')
        # self.set_param_hint('height', expr=f'{self.prefix}amplitude / ({self.prefix}sigma * {self.height_factor})') double check.

    def guess(self, data, x, negative=False, **kwargs):
        """
        Estimate initial parameter values from data.

        Parameters:
        - data (array): The dependent data.
        - x (array): The independent variable.
        - negative (bool): Whether to invert the data for peak finding.

        Returns:
        - Parameters: An lmfit.Parameters object with initial guesses.
        """
        # Guess the peak amplitude
        amplitude_guess = np.max(data) if not negative else np.min(data)
        # Guess the peak center
        center_guess = x[np.argmax(data)] if not negative else x[np.argmin(data)]
        # Guess sigma as a fraction of the total range
        sigma_guess = (x.max() - x.min()) / 6.0  # Rough estimate for sigma

        params = self.make_params(amplitude=amplitude_guess, center=center_guess, sigma=sigma_guess)
        return params


class CustomGaussian_ddx_Model(Model):
    """A custom model of the Gaussian 1st derivative lineshape copying the default lmfit GaussianModel behavior."""

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    height_factor = 1 / np.sqrt(2 * np.pi)

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        """
        Initialize the CustomGaussianModel.

        Parameters:
        - independent_vars (list): List of independent variable names.
        - prefix (str): Prefix for parameter names.
        - nan_policy (str): Handling of NaN values.
        """
        super().__init__(stable_gaussian_derivative_sigma, prefix=prefix, independent_vars=independent_vars,
                         nan_policy=nan_policy, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """Set parameter hints for amplitude, center, sigma, fwhm, and height."""
        self.set_param_hint(
            'amplitude')  # Allow negative for inverted a term?  Amplitude is probably stritly positive but we might be able to allow negative values for optimization reasons.
        self.set_param_hint('center')
        self.set_param_hint('sigma',
                            min=0)  # Sigma must be positive - can remove to allow the optimizer to range over greater space easily?

        # Expressions for derived parameters fwhm and height
        self.set_param_hint('fwhm', expr=f'{self.prefix}sigma * {self.fwhm_factor}')
        # self.set_param_hint('height', expr=f'{self.prefix}amplitude / ({self.prefix}sigma * {self.height_factor})') double check.

    def guess(self, data, x, negative=False, **kwargs):
        """
        Estimate initial parameter values from data.
        This is just code copying the default behavior from lmfit.
        We should never use this code because we have a sophisticated initial guess algo.

        Parameters:
        - data (array): The dependent data.
        - x (array): The independent variable.
        - negative (bool): Whether to invert the data for peak finding.

        Returns:
        - Parameters: An lmfit.Parameters object with initial guesses.
        """
        # Guess the peak amplitude
        amplitude_guess = np.max(data) if not negative else np.min(data)
        # Guess the peak center
        center_guess = x[np.argmax(data)] if not negative else x[np.argmin(data)]
        # Guess sigma as a fraction of the total range
        sigma_guess = (x.max() - x.min()) / 6.0  # Rough estimate for sigma

        params = self.make_params(amplitude=amplitude_guess, center=center_guess, sigma=sigma_guess)
        return params


# Function to fit Gaussians using lmfit with positive constraints for amplitude and sigma
def fit_gaussians(x, y, num_basis_gaussians, amplitudes, centers, sigmas):
    "the new fitting function using custom classes."
    model = None
    params = lmfit.Parameters()

    for i in range(num_basis_gaussians):
        g = CustomGaussianModel(prefix=f'g{i}_')
        if model is None:
            model = g
        else:
            model = model + g
        params.update(g.make_params())
        # Initialize parameters with bounds using `add()` method
        params.add(f'g{i}_center', value=centers[i], min=centers[i] - (centers[i] * PERCENT_RANGE_X / 100),
                   max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100), vary=VARY_CENTERS)  # Set bounds for center
        params.add(f'g{i}_amplitude', value=amplitudes[i],
                   min=0)  # min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE), max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE))           # Amplitude must be positive
        params.add(f'g{i}_sigma', value=sigmas[
            i])  # min=0, max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE))            # Sigma must be positive, example upper bound

        # Set center without bound - must either fix centers or very bounds by some small amount.
        # params[f'g{i}_center'].set(centers[i], vary=True)

        # Set amplitude and sigma with lower bounds to ensure they are positive
        # params[f'g{i}_amplitude'].set(amplitudes[i], min=0)  # Amplitude must be positive
        # params[f'g{i}_sigma'].set(sigmas[i])  # Sigma (width) must be positive

        # Set new parameter with bounds ±10% - this gives us the ability to "anneal" the fit, relax params - EVEN WHEN NO CURVES ARE REMOVED.
        # params.add(f'g{i}_center', value=centers[i],
        #                   min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100),
        #                   max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100),
        #                   vary=False)

        # params.add(f'g{i}_amplitude', value=amplitudes[i],
        #                   min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE / 100),
        #                   max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE / 100))

        # params.add(f'g{i}_sigma', value=sigmas[i],
        #                   min=sigmas[i] - (sigmas[i] * PERCENTAGE_RANGE / 100),
        #                   max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE / 100))

    result = model.fit(y, params, x=x)
    return result, model


# Function to fit Gaussians using lmfit with positive constraints for amplitude and sigma
def fit_gaussian_derivatives(x, y, num_basis_gaussians, amplitudes, centers, sigmas):
    "fitting gaussian derivatives using the new fitting function w/ custom classes."
    model = None
    params = lmfit.Parameters()

    for i in range(num_basis_gaussians):
        g = CustomGaussian_ddx_Model(prefix=f'g{i}_')
        if model is None:
            model = g
        else:
            model = model + g
        params.update(g.make_params())
        # Initialize parameters with bounds using `add()` method
        # Here, we relax the requirement that amplitude be positive.
        # Try playing wit constraints and normalization in the gaussian model. bookmark.

        # I think this is reasonable bc the direction of integration might matter?
        params.add(f'g{i}_center', value=centers[
            i])  # , min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100), max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100), vary=True)  # Set bounds for center
        params.add(f'g{i}_amplitude', value=amplitudes[
            i])  # min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE), max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE))           # Amplitude must be positive
        params.add(f'g{i}_sigma', value=sigmas[
            i])  # min=0, max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE))            # Sigma must be positive, example upper bound

        # Set center without bound - must either fix centers or very bounds by some small amount.
        # params[f'g{i}_center'].set(centers[i], vary=True)

        # Set amplitude and sigma with lower bounds to ensure they are positive
        # params[f'g{i}_amplitude'].set(amplitudes[i], min=0)  # Amplitude must be positive
        # params[f'g{i}_sigma'].set(sigmas[i])  # Sigma (width) must be positive

        # Set new parameter with bounds ±10% - this gives us the ability to "anneal" the fit, relax params - EVEN WHEN NO CURVES ARE REMOVED.
        # params.add(f'g{i}_center', value=centers[i],
        #                   min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100),
        #                   max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100),
        #                   vary=False)

        # params.add(f'g{i}_amplitude', value=amplitudes[i],
        #                   min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE / 100),
        #                   max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE / 100))

        # params.add(f'g{i}_sigma', value=sigmas[i],
        #                   min=sigmas[i] - (sigmas[i] * PERCENTAGE_RANGE / 100),
        #                   max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE / 100))

    result = model.fit(y, params, x=x)
    return result, model


# Function to fit Gaussians using lmfit with positive constraints for amplitude and sigma
def fit_gaussians_old(x, y, num_basis_gaussians, amplitudes, centers, sigmas):
    "the old fitting function before moving to the custom classes."
    model = None
    params = lmfit.Parameters()

    for i in range(num_basis_gaussians):
        g = GaussianModel(prefix=f'g{i}_')
        if model is None:
            model = g
        else:
            model = model + g
        params.update(g.make_params())
        # Initialize parameters with bounds using `add()` method
        params.add(f'g{i}_center', value=centers[
            i])  # , min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100), max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100), vary=True)  # Set bounds for center
        params.add(f'g{i}_amplitude', value=amplitudes[i],
                   min=0)  # min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE), max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE))           # Amplitude must be positive
        params.add(f'g{i}_sigma', value=sigmas[
            i])  # min=0, max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE))            # Sigma must be positive, example upper bound

        # Set center without bound - must either fix centers or very bounds by some small amount.
        # params[f'g{i}_center'].set(centers[i], vary=True)

        # Set amplitude and sigma with lower bounds to ensure they are positive
        # params[f'g{i}_amplitude'].set(amplitudes[i], min=0)  # Amplitude must be positive
        # params[f'g{i}_sigma'].set(sigmas[i])  # Sigma (width) must be positive

        # Set new parameter with bounds ±10% - this gives us the ability to "anneal" the fit, relax params - EVEN WHEN NO CURVES ARE REMOVED.
        # params.add(f'g{i}_center', value=centers[i],
        #                   min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100),
        #                   max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100),
        #                   vary=False)

        # params.add(f'g{i}_amplitude', value=amplitudes[i],
        #                   min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE / 100),
        #                   max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE / 100))

        # params.add(f'g{i}_sigma', value=sigmas[i],
        #                   min=sigmas[i] - (sigmas[i] * PERCENTAGE_RANGE / 100),
        #                   max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE / 100))

    result = model.fit(y, params, x=x)
    return result, model


# Function to generate multiple Gaussians with added noise
def generate_gaussians(x, num_gaussians=NUM_GAUSSIANS, noise_level=NOISE_LEVEL, seed=SEED, min_distance=MIN_DISTANCE):
    y = np.zeros_like(x)
    gaussians = []  # Store each Gaussian component
    centers = []  # To track centers of generated Gaussians
    np.random.seed(seed)  # For reproducibility

    for _ in range(num_gaussians):
        # Generate a random center and ensure it's at least `min_distance` away from existing centers
        while True:
            center = np.random.uniform(x.min(), x.max())
            if all(abs(center - c) >= min_distance for c in centers):
                centers.append(center)
                break

        # Generate random amplitude and width for the Gaussian
        amplitude = np.random.uniform(0.5, 1.5)
        width = np.random.uniform(1, 3)

        # Create the Gaussian and add it to the total signal
        g = gaussian(x, amplitude, center, width)
        gaussians.append(g)
        y += g

    # Add white noise
    y += np.random.normal(0, noise_level, size=x.shape)

    return y, gaussians


# Function to estimate sigma using FWHM
def estimate_sigma(x, y, peak_index):
    half_max = y[peak_index] / 2
    left_candidates = np.where(y[:peak_index] < half_max)[0]
    if len(left_candidates) == 0:
        left_idx = 0  # If no valid left index, use the start of the array
    else:
        left_idx = left_candidates[-1]

    right_candidates = np.where(y[peak_index:] < half_max)[0]
    if len(right_candidates) == 0:
        right_idx = len(y) - 1  # If no valid right index, use the end of the array
    else:
        right_idx = right_candidates[0] + peak_index

    fwhm = x[right_idx] - x[left_idx]
    return abs(fwhm / SMALL_FWHM_FACTOR)  # Convert FWHM to sigma


# Function to generate initial guesses for Gaussian parameters
def generate_initial_guesses(x, y, num_gaussians):
    # Smooth the noisy data
    y_smoothed = savgol_filter(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Calculate the numerical derivatives

    d_y_smoothed = np.gradient(y_smoothed, x)
    # Calculate the 2nd numerical derivatives
    dd_y = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = savgol_filter(dd_y_smoothed, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Find peaks in the negative second derivative (to locate the centers of Gaussians)
    prominence = PROMINENCE_PERECENT * max(dd_y)
    height = HEIGHT_THRESHOLD * max(dd_y)
    dd_y_peaks, _ = find_peaks(-dd_y_smoothed, height=height, distance=DISTANCE, prominence=prominence)

    peak_centers = x[dd_y_peaks]
    peak_amplitudes = y_smoothed[dd_y_peaks]
    # this would work if my gaussian is normalized to unit height. lets try writing this so that we are normalized to unit area. brb
    peak_sigmas = [estimate_sigma(x, y_smoothed, peak) for peak in dd_y_peaks]
    # estimating sigma from raw data is troublesome. Consider trying to do so from second derivative or solve analytically using peak height. Of course, the derivative would need to be normalzied.

    # If identified more peaks than needed, sort by amplitude and keep the strongest ones
    if len(peak_centers) > num_gaussians:
        sorted_indices = np.argsort(peak_amplitudes)[-num_gaussians:]
        peak_centers = peak_centers[sorted_indices]
        peak_amplitudes = peak_amplitudes[sorted_indices]
        peak_sigmas = np.array(peak_sigmas)[sorted_indices]

    print(f'Initial Guess Peak Centers: {peak_centers}')
    print(f'Initial Guess Peak Sigmas: {peak_sigmas}')
    print(f'Intial Guess Peak Amplitudes: {peak_amplitudes}')

    # Plot the true combined Gaussian curve and smoothed curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='True Combined Gaussian Curve', color='blue', linewidth=2)
    plt.plot(x, y_smoothed, label='Smoothed Combined Curve (y_smoothed)', color='green', linestyle='-', linewidth=2)

    # Plot the numerical derivatives of the smoothed curve
    plt.plot(x, -dd_y_smoothed, label='-1 * Numerical Derivative of y_smoothed', color='green', linestyle='--',
             linewidth=2)
    plt.scatter(x[dd_y_peaks], -dd_y_smoothed[dd_y_peaks], color='red', label='Detected Peaks')

    # Plot the initial guesses for each Gaussian curve
    for center, amplitude, sigma in zip(peak_centers, peak_amplitudes, peak_sigmas):
        gaussian_guess = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        plt.plot(x, gaussian_guess, label=f'Gaussian guess: Center={center:.2f}, Sigma={sigma:.2f}', linestyle='--')

    # Add title and labels
    plt.title('True Combined Gaussian Curve, Smoothed Curve, and Their Derivatives')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the numerical derivatives of the smoothed curve
    plt.plot(x, -dd_y_smoothed, label='-1 * Numerical Derivative of y_smoothed', color='green', linestyle='--',
             linewidth=2)
    plt.scatter(x[dd_y_peaks], -dd_y_smoothed[dd_y_peaks], color='red', label='Detected Peaks')
    plt.show()

    return peak_amplitudes, peak_centers, peak_sigmas


def generate_initial_guesses_A(x, y, num_gaussians):
    # Smooth the noisy data
    y_smoothed = savgol_filter(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Calculate the numerical derivatives
    d_y = np.gradient(y_smoothed, x)
    d_y_smoothed = savgol_filter(d_y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Find peaks in the negative second derivative (to locate the centers of Gaussians)
    prominence = PROMINENCE_PERECENT * max(d_y)
    d_y_peaks, _ = find_peaks(-d_y_smoothed, height=HEIGHT_THRESHOLD, distance=DISTANCE, prominence=prominence)

    peak_centers = x[d_y_peaks]
    peak_amplitudes = y_smoothed[d_y_peaks]  # this wont work for a terms
    # this would work if my gaussian is normalized to unit height.
    # going to need to get that special gaussian going. brb
    # might need to modify this to unit area and integrate.
    peak_sigmas = [estimate_sigma(x, y_smoothed, peak) for peak in d_y_peaks]
    # estimating sigma from raw data is troublesome. Consider trying to do so from second derivative or solve analytically using peak height. Of course, the derivative would need to be normalzied.

    # If identified more peaks than needed, sort by amplitude and keep the strongest ones
    if len(peak_centers) > num_gaussians:
        sorted_indices = np.argsort(peak_amplitudes)[-num_gaussians:]
        peak_centers = peak_centers[sorted_indices]
        peak_amplitudes = peak_amplitudes[sorted_indices]
        peak_sigmas = np.array(peak_sigmas)[sorted_indices]

    print(f'Initial Guess Peak Centers: {peak_centers}')
    print(f'Initial Guess Peak Sigmas: {peak_sigmas}')
    print(f'Intial Guess Peak Amplitudes: {peak_amplitudes}')

    # Plot the true combined Gaussian curve and smoothed curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='True Combined Gaussian Curve', color='blue', linewidth=2)
    plt.plot(x, y_smoothed, label='Smoothed Combined Curve (y_smoothed)', color='green', linestyle='-', linewidth=2)

    # Plot the numerical derivatives of the smoothed curve
    plt.plot(x, -d_y_smoothed, label='-1 * Numerical Derivative of y_smoothed', color='green', linestyle='--',
             linewidth=2)
    plt.scatter(x[d_y_peaks], -d_y_smoothed[d_y_peaks], color='red', label='Detected Peaks')

    # Plot the initial guesses for each Gaussian curve
    for center, amplitude, sigma in zip(peak_centers, peak_amplitudes, peak_sigmas):
        gaussian_guess = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        plt.plot(x, gaussian_guess, label=f'Gaussian guess: Center={center:.2f}, Sigma={sigma:.2f}', linestyle='--')

    # Add title and labels
    plt.title('True Combined Gaussian Curve, Smoothed Curve, and Their Derivatives')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the numerical derivatives of the smoothed curve
    plt.plot(x, -d_y_smoothed, label='-1 * Numerical Derivative of y_smoothed', color='green', linestyle='--',
             linewidth=2)
    plt.scatter(x[d_y_peaks], -d_y_smoothed[d_y_peaks], color='red', label='Detected Peaks')
    plt.show()

    return peak_amplitudes, peak_centers, peak_sigmas


# Function to remove least impactful Gaussians by contribution to RSS
def remove_least_impactful_gaussian_derivatives_by_fit(x, y, result, num_basis, rss_threshold_percent=5):
    rss_increases = []
    least_impactful_gaussians = []

    original_rss = np.sum((y - result.best_fit) ** 2)  # Original RSS

    # Loop over each Gaussian and compute the RSS increase if removed
    for i in range(num_basis):
        reduced_model = None
        reduced_params = lmfit.Parameters()

        # Build a model excluding Gaussian i
        for j in range(num_basis):
            if j == i:
                continue  # Skip the Gaussian being tested for removal
            param_prefix = f'g{j}_'
            if any(param_prefix in key for key in result.params.keys()):
                g = CustomGaussian_ddx_Model(prefix=param_prefix)
                if reduced_model is None:
                    reduced_model = g
                else:
                    reduced_model = reduced_model + g
                reduced_params.add(f'{param_prefix}center', value=result.params[f'{param_prefix}center'].value,
                                   vary=VARY_CENTERS)
                reduced_params.add(f'{param_prefix}amplitude', value=result.params[f'{param_prefix}amplitude'].value,
                                   vary=False)
                reduced_params.add(f'{param_prefix}sigma', value=result.params[f'{param_prefix}sigma'].value,
                                   vary=False)

        if reduced_model:
            reduced_result = reduced_model.fit(y, reduced_params, x=x)
            reduced_rss = np.sum((y - reduced_result.best_fit) ** 2)
            rss_increase = reduced_rss - original_rss
            rss_increases.append(rss_increase)

            print(f"Removing Gaussian {i} increases RSS by {rss_increase:.4f}")

    # Calculate total sum of RSS increases
    total_rss_increase = sum(rss_increases)

    # If no total increase (or all zeros), avoid division by zero
    if total_rss_increase == 0:
        print("Total RSS increase is zero, no significant impact detected.")
        return least_impactful_gaussians

    # Calculate percentage increase for each Gaussian
    for i, rss_increase in enumerate(rss_increases):
        percentage_increase = (rss_increase / total_rss_increase) * 100
        print(f"Gaussian {i}: RSS increase contribution = {percentage_increase:.2f}%")

        # Append only the Gaussians that have a percentage increase below the threshold
        if percentage_increase < rss_threshold_percent:
            least_impactful_gaussians.append((i, percentage_increase))

    # Sort least impactful Gaussians by percentage (optional)
    least_impactful_gaussians.sort(key=lambda x: x[1])
    print(f'Least impactful gaussians: {least_impactful_gaussians}')

    return least_impactful_gaussians


def remove_least_impactful_gaussians_by_fit(x, y, result, num_basis, rss_threshold_percent=5):
    rss_increases = []
    least_impactful_gaussians = []

    original_rss = np.sum((y - result.best_fit) ** 2)  # Original RSS

    # Loop over each Gaussian and compute the RSS increase if removed
    for i in range(num_basis):
        reduced_model = None
        reduced_params = lmfit.Parameters()

        # Build a model excluding Gaussian i
        for j in range(num_basis):
            if j == i:
                continue  # Skip the Gaussian being tested for removal
            param_prefix = f'g{j}_'
            if any(param_prefix in key for key in result.params.keys()):
                g = GaussianModel(prefix=param_prefix)
                if reduced_model is None:
                    reduced_model = g
                else:
                    reduced_model = reduced_model + g
                reduced_params.add(f'{param_prefix}center', value=result.params[f'{param_prefix}center'].value,
                                   vary=VARY_CENTERS)  # think we need to not vary these so as to get a good idea of how much each contributes
                reduced_params.add(f'{param_prefix}amplitude', value=result.params[f'{param_prefix}amplitude'].value,
                                   vary=False)
                reduced_params.add(f'{param_prefix}sigma', value=result.params[f'{param_prefix}sigma'].value,
                                   vary=False)

        if reduced_model:
            reduced_result = reduced_model.fit(y, reduced_params, x=x)
            reduced_rss = np.sum((y - reduced_result.best_fit) ** 2)
            rss_increase = reduced_rss - original_rss
            rss_increases.append(rss_increase)

            print(f"Removing Gaussian {i} increases RSS by {rss_increase:.4f}")

    # Calculate total sum of RSS increases
    total_rss_increase = sum(rss_increases)

    # If no total increase (or all zeros), avoid division by zero
    if total_rss_increase == 0:
        print("Total RSS increase is zero, no significant impact detected.")
        return least_impactful_gaussians

    # Calculate percentage increase for each Gaussian
    for i, rss_increase in enumerate(rss_increases):
        percentage_increase = (rss_increase / total_rss_increase) * 100
        print(f"Gaussian {i}: RSS increase contribution = {percentage_increase:.2f}%")

        # Append only the Gaussians that have a percentage increase below the threshold
        if percentage_increase < rss_threshold_percent:
            least_impactful_gaussians.append((i, percentage_increase))

    # Sort least impactful Gaussians by percentage (optional)
    least_impactful_gaussians.sort(key=lambda x: x[1])
    print(f'Least impactful gaussians: {least_impactful_gaussians}')

    return least_impactful_gaussians


# Function to compute and plot the numerical derivative of the "true" combined Gaussian curve
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


def iterate_and_fit_gaussians(x, y, z, max_basis_gaussians=MAX_BASIS_GAUSSIANS, num_guesses=NUM_GUESSES):
    # we passsed the gaussians to this function so that we can visualize them, but they arent required for fitting obvs.
    avg_bic_values = []
    avg_delta_bic_values = []
    all_fits = []  # To store all iterations of N-1 fits

    amplitudes, centers, sigmas = generate_initial_guesses(x, y, max_basis_gaussians)
    previous_bic = None  # Make sure that this initilizes to None so that we dont have memory issues.
    lowest_delta_bic = float('inf')  # Initialize to a large value
    lowest_delta_bic_idx = -1  # Index for N with lowest delta BIC - here we are going to lowest bic then going back one step?
    lowest_bic = float('inf')
    lowest_bic_idx = -1  # Index for N with lowest delta BIC - not exactly sure here?

    # Fit Gaussians for different numbers of basis functions
    # for num_basis in range(1, max_basis_gaussians + 1):
    # but cant fit more curves than we have guesses.
    for num_basis in range(1, len(centers) + 1):
        bic_list = []
        fits = []
        # Try multiple guesses -
        for guess in range(num_guesses):
            result, model = fit_gaussians(x, y, num_basis, amplitudes[:num_basis], centers[:num_basis],
                                          sigmas[:num_basis])
            rss = np.sum((y - result.best_fit) ** 2)
            # num_params = 3 * num_basis # because we have mu, gamma, amp? was used to calc bic but dont need
            bic = result.bic
            print(f'result.bic:{bic}')
            bic_list.append(bic)
            fits.append(result)

            avg_bic = np.mean(bic_list)  # because I assume I am taking multiple guesses - but I have been limiting number of guesses to 1 so the average of 1 number is itself.
            avg_bic_values.append(avg_bic)

        # Calculate delta BIC
        if previous_bic is not None:
            delta_bic = avg_bic - previous_bic  # This is not delta bic by itself... ?
            avg_delta_bic_values.append(delta_bic)
            print(f'num_basis: {num_basis} | avg BIC = {avg_bic} | delta BIC: {delta_bic}')

            # Check for lowest BIC value
            if avg_bic < lowest_bic:
                lowest_bic = avg_bic
                lowest_bic_idx = num_basis

            # Early stop if delta BIC is below threshold
            if abs(delta_bic) < DELTA_BIC_THRESHOLD:
                print(f"Delta BIC ~ 0 at N = {num_basis}. Reporting fits at N-1 = {num_basis - 1}.")
                all_fits = fits
                break

        previous_bic = avg_bic

    # If no early stop occurred, return the fit with the lowest BIC
        if not all_fits:
            if lowest_bic_idx > 1:
                print(f"Returning the N = {lowest_bic_idx} basis functions fit where BIC was minimized.")
                num_basis = lowest_bic_idx
            else:
                num_basis = 1
            all_fits = fits

    # all_fits = fits

    # Plot all iterations of the fits with N Gaussians, including true Gaussians. bookmark here
    for fit_idx, fit in enumerate(all_fits):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'b', label='Data with noise', zorder=1)

        # Plot the true original Gaussians - cant in real data
        # for i, g_true in enumerate(gaussians):
        #    plt.plot(x, g_true, 'g-', label=f'True Gaussian {i}', alpha=0.7, zorder=2)

        # Plot the individual fitted Gaussian components for this fit
        for i in range(num_basis):
            g_fit = fit.eval_components()[f'g{i}_']
            plt.plot(x, g_fit, label=f'Fitted Gaussian {i} in Fit {fit_idx}', linestyle='--', zorder=4)

        # Plot the composite fit
        plt.plot(x, fit.best_fit, 'r-', label=f'Composite Fit {fit_idx}', zorder=3)
        plt.title(f'Constituent and True Gaussians of N, Iteration: {lowest_bic_idx}, Fit: {fit_idx}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    # Identify the least impactful Gaussians and remove them
    impactful_gaussians = remove_least_impactful_gaussians_by_fit(x, y, result, num_basis,
                                                                  rss_threshold_percent=THRESHOLD_PERCENT)
    print(f"Least impactful Gaussians are: {impactful_gaussians}")

    # Extract indices of least impactful Gaussians
    impactful_gaussian_indices = [i for i, _ in impactful_gaussians]

    # Re-fit the model without the least impactful Gaussians
    print(f"Re-fitting after removing Gaussians: {impactful_gaussian_indices}...")
    reduced_model = None
    reduced_params = lmfit.Parameters()

    # Build the model excluding the least impactful Gaussians
    for i in range(num_basis):
        if i in impactful_gaussian_indices:
            continue  # Skip the least impactful Gaussians

        g = CustomGaussianModel(prefix=f'g{i}_')  # did this change anything by chaning away from default?
        if reduced_model is None:
            reduced_model = g
        else:
            reduced_model = reduced_model + g

        # Get current parameter values
        center_value = result.params[f'g{i}_center'].value
        amplitude_value = result.params[f'g{i}_amplitude'].value
        sigma_value = result.params[f'g{i}_sigma'].value

        # Set new parameter with bounds ±10% - this gives us the ability to "anneal" the fit, relax params - EVEN WHEN NO CURVES ARE REMOVED.
        reduced_params.add(f'g{i}_center', value=center_value,
                           min=center_value - (center_value * PERCENT_RANGE_X / 100),
                           max=center_value + (center_value * PERCENT_RANGE_X / 100))

        reduced_params.add(f'g{i}_amplitude', value=amplitude_value,
                           min=amplitude_value - (amplitude_value * PERCENTAGE_RANGE / 100),
                           max=amplitude_value + (amplitude_value * PERCENTAGE_RANGE / 100))

        reduced_params.add(f'g{i}_sigma', value=sigma_value,
                           min=sigma_value - (sigma_value * PERCENTAGE_RANGE / 100),
                           max=sigma_value + (sigma_value * PERCENTAGE_RANGE / 100))

    # Perform the fit again after removing the least impactful Gaussians
    reduced_result = reduced_model.fit(y, reduced_params, x=x)

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

    ### Use fit values as initial guesses for A term fitting. ###

    A_model = None
    A_params = lmfit.Parameters()

    # Use only the indices of the Gaussians that were not removed
    remaining_indices = [i for i in range(num_basis) if i not in impactful_gaussian_indices]

    # for i in range(num_basis): # no longer
    for i in remaining_indices:
        g = CustomGaussian_ddx_Model(prefix=f'g{i}_')
        if A_model is None:
            A_model = g
        else:
            A_model = A_model + g
        A_params.update(g.make_params())

        # Get current parameter values
        center_value = reduced_result.params[f'g{i}_center'].value
        amplitude_value = reduced_result.params[f'g{i}_amplitude'].value
        sigma_value = reduced_result.params[f'g{i}_sigma'].value

        A_params.add(f'g{i}_center', value=center_value,
                     vary=VARY_CENTERS)  # , min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100), max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100), vary=True)  # Set bounds for center
        A_params.add(f'g{i}_amplitude',
                     value=amplitude_value)  # min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE), max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE))           # Amplitude must be positive
        A_params.add(f'g{i}_sigma', value=sigma_value)

        # Set new parameter with bounds ±10% - this gives us the ability to "anneal" the fit, relax params - EVEN WHEN NO CURVES ARE REMOVED.
        # params.add(f'g{i}_center', value=centers[i],
        #                   min=centers[i] - (centers[i] * PERCENTAGE_RANGE / 100),
        #                   max=centers[i] + (centers[i] * PERCENTAGE_RANGE / 100),
        #                   vary=False)

        # params.add(f'g{i}_amplitude', value=amplitudes[i],
        #                   min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE / 100),
        #                   max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE / 100))

        # params.add(f'g{i}_sigma', value=sigmas[i],
        #                   min=sigmas[i] - (sigmas[i] * PERCENTAGE_RANGE / 100),
        #                   max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE / 100))

    print(f"Fitting A-terms with initial guesses: {A_params}...")
    A_result = A_model.fit(z, A_params, x=x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, z, 'b', label='Data with noise', zorder=1)

    # Plot the individual fitted Gaussian derivative components for this fit
    for i in remaining_indices:
        dg_fit = A_result.eval_components()[f'g{i}_']
        plt.plot(x, dg_fit, label=f'Fitted Gaussian {i}', linestyle='--', zorder=3)

    # Plot the composite fit
    plt.plot(x, A_result.best_fit, 'r-', label=f'Composite Fit', zorder=2)
    plt.title(f'MCD initial fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    ## Remove the least impactful A-term fits. If buggy, may have to use different variable names.

    # Identify the least impactful Gaussians and remove them
    impactful_gaussian_derivatives = remove_least_impactful_gaussian_derivatives_by_fit(x, z, A_result,
                                                                                        len(remaining_indices),
                                                                                        rss_threshold_percent=THRESHOLD_PERCENT)
    print(f"Least impactful Gaussians are: {impactful_gaussian_derivatives}")

    # Extract indices of least impactful Gaussians
    impactful_gaussian_derivative_indices = [i for i, _ in impactful_gaussian_derivatives]

    # Re-fit the model without the least impactful Gaussians
    print(f"Re-fitting after removing Gaussians: {impactful_gaussian_derivative_indices}...")
    reduced_derivative_model = None
    reduced_derivative_params = lmfit.Parameters()

    remaining_derivative_indices = [i for i in remaining_indices if i not in impactful_gaussian_derivative_indices]

    # Build the model excluding the least impactful Gaussians
    for i in remaining_derivative_indices:
        # if i in impactful_gaussian_derivative_indices: # dont need this anymore
        #    continue  # Skip the least impactful Gaussians

        g = CustomGaussian_ddx_Model(prefix=f'g{i}_')  # need to change to gaussian derivative
        if reduced_derivative_model is None:
            reduced_derivative_model = g
        else:
            reduced_derivative_model = reduced_derivative_model + g

        # Get current parameter values
        derivative_center_value = A_result.params[f'g{i}_center'].value
        derivative_amplitude_value = A_result.params[f'g{i}_amplitude'].value
        derivative_sigma_value = A_result.params[f'g{i}_sigma'].value

        # Set new parameter with bounds ±10% - this gives us the ability to "anneal" the fit, relax params - EVEN WHEN NO CURVES ARE REMOVED.
        reduced_derivative_params.add(f'g{i}_center', value=derivative_center_value, vary=VARY_CENTERS,
                                      min=derivative_center_value - (derivative_center_value * PERCENTAGE_RANGE / 100),
                                      max=derivative_center_value + (derivative_center_value * PERCENTAGE_RANGE / 100))

        reduced_derivative_params.add(f'g{i}_amplitude', value=derivative_amplitude_value,
                                      min=derivative_amplitude_value - (
                                                  derivative_amplitude_value * PERCENTAGE_RANGE / 100),
                                      max=derivative_amplitude_value + (
                                                  derivative_amplitude_value * PERCENTAGE_RANGE / 100))

        reduced_derivative_params.add(f'g{i}_sigma', value=derivative_sigma_value,
                                      min=derivative_sigma_value - (derivative_sigma_value * PERCENTAGE_RANGE / 100),
                                      max=derivative_sigma_value + (derivative_sigma_value * PERCENTAGE_RANGE / 100))

    # Perform the fit again after removing the least impactful Gaussians
    reduced_A_result = reduced_derivative_model.fit(z, reduced_derivative_params, x=x)

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

    # Report the parameters of interest.
    print(f'A-Term Parameters: {reduced_A_result.params}')
    print(f'Electronic Dipole Parameters: {reduced_result.params}')

    # Extract parameters from the reduced_A_result and reduced_result
    aterm_params = reduced_A_result.params
    dipole_params = reduced_result.params

    # Pass the x-values (wavenumbers) to calculate the range
    x_values = mcd_df['wavenumber'].values

    # Group centers and calculate A/D ratio, using a percentage of the total x range for tolerance
    grouped_data = group_centers(dipole_params, aterm_params, x_values, tolerance_percentage=TOLERANCE_X)

    # Create a DataFrame to display the result
    df = pd.DataFrame(grouped_data)

    # Display the formatted table
    print(df)

    return df

    # Optionally, display the table with improved formatting
    # df.style.format({
    #    'Dipole Center (D)': '{:.2f}',
    #    'A-Term Center (A)': '{:.2f}',
    #    'Amplitude (D)': '{:.4f}',
    #    'Amplitude (A)': '{:.4f}',
    #    'A/D Ratio': '{:.4f}'
    # })

    # print(df)


# Main Execution (Reading from CSV)
if __name__ == "__main__":
    file_path = "/Users/westo/Downloads/PtTrans/PtTrans/processed_data/PtTrans_processed.csv"
    file_directory = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace("__processed.csv", "")
    save_file_path = os.path.join(file_directory, f"{base_name}_A_term.csv")

    mcd_df = pd.read_csv(file_path)

    # Prepare the dataframe
    mcd_df['wavenumber'] = 1e7 / mcd_df['wavelength']
    mcd_df['scaled_absorption'] = mcd_df['intensity'] / (mcd_df['wavenumber'] * 1.315 * 326.6)
    mcd_df['scaled_MCD'] = mcd_df['R_signed'] / (mcd_df[
                                                                'wavenumber'] * 1.315 * 152.5)  # Is this even orientational averaging? I get reasonable values if I dont do the orientational averaging for MCD.

    plt.plot(mcd_df['wavenumber'], mcd_df['intensity'])
    plt.show()
    plt.plot(mcd_df['wavenumber'], mcd_df['R_signed'])  # To make sure things are scaling.
    plt.show()

    # Extract x and y values
    wavenumbers = mcd_df['wavenumber'].values
    scaled_absorption = mcd_df['scaled_absorption'].values
    scaled_mcd = mcd_df['scaled_MCD'].values

    # Perform Gaussian fitting on the data from the CSV
    df = iterate_and_fit_gaussians(wavenumbers, scaled_absorption, scaled_mcd)

    print(f'Saving output to: {save_file_path}')
    # Save the DataFrame (df) to the new CSV file
    df.to_csv(save_file_path, index=False)  # 'index=False' avoids saving the row numbers as an extra column