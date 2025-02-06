import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import GaussianModel
import lmfit
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
import os
from . import data_plotting as dplt
from .CustomGaussianModel import CustomGaussianModel
from .CustomGaussian_ddx_Model import CustomGaussian_ddx_Model
from .constants import *
from .gaussians import gaussian
from .utils import file_handler as fh

# Define all the constants



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


# Function to generate a single Gaussian


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
        params.add(f'g{i}_amplitude', value=amplitudes[i],min=0.0)  # min=amplitudes[i] - (amplitudes[i] * PERCENTAGE_RANGE), max=amplitudes[i] + (amplitudes[i] * PERCENTAGE_RANGE))           # Amplitude must be positive
        params.add(f'g{i}_sigma', value=sigmas[i], max=MAX_SIGMA)  # min=0, max=sigmas[i] + (sigmas[i] * PERCENTAGE_RANGE))            # Sigma must be positive, example upper bound
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

#Function for finding some width at some ratio of max from peak - similar to FWHM
def get_anymax_factor(ratio):
    if (ratio >= 1): #return FWHM if ratio is invalid
        print("full width any max has invalid ratio")
        return SMALL_FWHM_FACTOR
    else:
        return np.sqrt(8*np.log(1/ratio))
# Function to estimate sigma
def estimate_sigma(x, y, peak_index, ratio):
    some_max = y[peak_index] * ratio
    left_candidates = np.where(y[:peak_index] < some_max)[0]
    if len(left_candidates) == 0:
        left_idx = 0  # If no valid left index, use the start of the array
    else:
        left_idx = left_candidates[-1]

    right_candidates = np.where(y[peak_index:] < some_max)[0]
    if len(right_candidates) == 0:
        right_idx = len(y) - 1  # If no valid right index, use the end of the array
    else:
        right_idx = right_candidates[0] + peak_index
    swsm = x[right_idx] - x[left_idx]
    sigma = abs(swsm / get_anymax_factor(ratio)) # Convert to sigma
    #cap sigma
    return min(sigma, MAX_SIGMA)
#estimates sigma for 1/10 max, 2/10 max, 3/10 max... up to 9/10 max
def estimate_average_sigma(x, y, peak_index):
    total = 0
    for i in range(1, ESTIMATE_SIGMA_ITERATIONS - 1):
        total += estimate_sigma(x,y,peak_index, i/ESTIMATE_SIGMA_ITERATIONS)
    return total / (ESTIMATE_SIGMA_ITERATIONS - 1)
# Function to generate initial guesses for Gaussian parameters
def generate_initial_guesses(x, y, num_gaussians):
    # Smooth the noisy data
    y_smoothed = savgol_filter(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    #print(y_smoothed)
    # Calculate the numerical derivatives
    d_y_smoothed = np.gradient(y_smoothed,x)
    # Calculate the 2nd numerical derivatives
    dd_y = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = savgol_filter(dd_y_smoothed, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    # Find peaks in the negative second derivative (to locate the centers of Gaussians)
    prominence = PROMINENCE_PERECENT * np.nanmax(dd_y)
    height = HEIGHT_THRESHOLD * np.nanmax(dd_y)
    #print(np.nanmax(dd_y))

    #interpolate so find peaks behaves right
    #A = np.interp(np.arange(len(dd_y_smoothed)),np.arange(len(dd_y_smoothed))[np.isnan(dd_y_smoothed) == False],dd_y_smoothed[np.isnan(dd_y_smoothed) == False])
    #print(dd_y_smoothed)
    #print(A)



    dd_y_peaks_all, _ = find_peaks(-dd_y_smoothed, height=height, distance=DISTANCE, prominence=prominence)
    print(dd_y_smoothed)
    dd_y_peaks = filter_peaks_deltax(x, dd_y_peaks_all)
    peak_centers = x[dd_y_peaks]
    peak_amplitudes = y_smoothed[dd_y_peaks]
    # this would work if my gaussian is normalized to unit height. lets try writing this so that we are normalized to unit area. brb
    peak_sigmas = [estimate_average_sigma(x, y_smoothed, peak) for peak in dd_y_peaks]
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

    dplt.plot_true_combined_and_smoothed(x,y,y_smoothed,dd_y_smoothed, dd_y_peaks, peak_centers, peak_amplitudes, peak_sigmas)

    return peak_amplitudes, peak_centers, peak_sigmas


def generate_initial_guesses_A(x, y, num_gaussians):
    # Smooth the noisy data
    y_smoothed = savgol_filter(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Calculate the numerical derivatives
    d_y = np.gradient(y_smoothed, x)
    d_y_smoothed = savgol_filter(d_y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Find peaks in the negative second derivative (to locate the centers of Gaussians)
    prominence = PROMINENCE_PERECENT * max(d_y)

    d_y_peaks_all, _ = find_peaks(-d_y_smoothed, height=HEIGHT_THRESHOLD, distance=DISTANCE, prominence=prominence)
    d_y_peaks = filter_peaks_deltax(x, d_y_peaks_all) #filter peaks by max peak delta x

    peak_centers = x[d_y_peaks]
    peak_amplitudes = y_smoothed[d_y_peaks]  # this wont work for a terms
    # this would work if my gaussian is normalized to unit height.
    # going to need to get that special gaussian going. brb
    # might need to modify this to unit area and integrate.
    peak_sigmas = [estimate_average_sigma(x, y_smoothed, peak) for peak in d_y_peaks]
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
    dplt.plot_true_combined_and_smoothed(x,y,y_smoothed,d_y_smoothed, d_y_peaks, peak_centers, peak_amplitudes, peak_sigmas)

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
                                   vary=False, max=MAX_SIGMA)

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

def filter_peaks_deltax(x, peaks):
    peak_list = list(peaks)
    center_prev = x[peaks[0]] #last center because of ordering
    prev_peak = peaks[0]
    #every peak but the first
    for peak in peaks[1:]:
        center = x[peak]
        if center_prev-center < MIN_PEAK_X_DISTANCE:
            peak_list.remove(peak)
            if prev_peak in peak_list:
                peak_list.remove(prev_peak)
        center_prev = center
        prev_peak = peak
    return np.array(peak_list)


def iterate_and_fit_gaussians(x, y, z, mcd_df, max_basis_gaussians=MAX_BASIS_GAUSSIANS, num_guesses=NUM_GUESSES):
    #remove NAN to avoid conflics with lmfit

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
    fits = []
    #Center length is zero
    #print(len(centers))
    for num_basis in range(1, len(centers) + 1):
        bic_list = []
        fits = []

        # Try multiple guesses -
        for guess in range(num_guesses):
            #print(sigmas[:num_basis])
            result, model = fit_gaussians(x, y, num_basis, amplitudes[:num_basis], centers[:num_basis],sigmas[:num_basis])
            #rss = np.sum((y - result.best_fit) ** 2)
            # num_params = 3 * num_basis # because we have mu, gamma, amp? was used to calc bic but dont need
            bic = result.bic
            print(f'result.bic:{bic}')
            bic_list.append(bic)
            fits.append(result)
            print(len(fits)+1)

        avg_bic = np.mean(
            bic_list)  # because I assume I am taking multiple guesses - but I have been limiting number of guesses to 1 so the average of 1 number is itself.
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
    # Plot all iterations of the fits with N Gaussians, including true Gaussians. bookmark here
    dplt.plot_gaussian_iterations(x,y,all_fits, num_basis, lowest_bic_idx)

    # Identify the least impactful Gaussians and remove them
    #float is being truncated by converting to int. needs to be int
    impactful_gaussians = remove_least_impactful_gaussians_by_fit(x, y, all_fits[-1], num_basis)
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
        A_params.add(f'g{i}_sigma', value=sigma_value, max=MAX_SIGMA)

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
        #print(A_result.params)
        #Dont display terms with super large sigma
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
    #plot
    dplt.plot_xz_after_gaussian_removal(x,z,reduced_A_result, remaining_derivative_indices, impactful_gaussian_derivative_indices)

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
def main():
    file_path = fh.select_processed_file()
    file_directory = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace("__processed.csv", "")
    save_file_path = os.path.join(file_directory, f"{base_name}_A_term.csv")

    mcd_df = pd.read_csv(file_path)

    # Prepare the dataframe
    mcd_df['wavenumber'] = 1e7 / mcd_df['wavelength']
    mcd_df['scaled_absorption'] = mcd_df['intensity'] / (mcd_df['wavenumber'] * 1.315 * 326.6)
    mcd_df['scaled_MCD'] = mcd_df['R_signed'] / (mcd_df['wavenumber'] * 1.315 * 152.5)  # Is this even orientational averaging? I get reasonable values if I dont do the orientational averaging for MCD.

    plt.plot(mcd_df['wavenumber'], mcd_df['intensity'])
    plt.show()
    plt.plot(mcd_df['wavenumber'], mcd_df['R_signed'])  # To make sure things are scaling.
    plt.show()

    # Extract x and y values
    wavenumbers = mcd_df['wavenumber'].values
    scaled_absorption = mcd_df['scaled_absorption'].values
    scaled_mcd = mcd_df['scaled_MCD'].values


    #remove data that corresponds with NAN
    mask = ~np.isnan(wavenumbers) & ~np.isnan(scaled_absorption) & ~np.isnan(scaled_mcd)
    x = wavenumbers[mask]
    y = scaled_absorption[mask]
    z = scaled_mcd[mask]

    # Perform Gaussian fitting on the data from the CSV
    df = iterate_and_fit_gaussians(x, y, z, mcd_df)

    print(f'Saving output to: {save_file_path}')
    # Save the DataFrame (df) to the new CSV file
    df.to_csv(save_file_path, index=False)  # 'index=False' avoids saving the row numbers as an extra column