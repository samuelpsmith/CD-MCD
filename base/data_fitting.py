#
# Module to hold all methods related to data fitting
#
import scipy.signal
from lmfit.models import GaussianModel
import lmfit
from scipy.signal import find_peaks, savgol_filter, peak_prominences
import pandas as pd
from . import data_plotting as dplt
from .CustomGaussianModel import CustomGaussianModel
from .CustomGaussian_ddx_Model import CustomGaussian_ddx_Model
from .constants import *

#Params: Dict dipole_params - parameters for the dipole
#        Dict aterm_params - parameters for the aterm
#        Dict x_values - x axis values
#        int tolerance_percentage - Tolerance as percentage of x range
#Retruns: List of dicts for grouped params
#Does: Groups close centers
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
#Params: numpy.ndarray x - x values
#        numpy.ndarray y - y values
#        int num_basis_gaussians - number of basis gaussians to fit
#        list amplitudes - List of the amplitudes of the gaussians
#        list centers - List of the centers of the gaussians
#        list sigmas - List of the std deviations of the gaussians
#Returns: pair of lmfit result and model
#Does: Function to fit gaussians using lmfit with positive constraints for amplitude and sigma
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
    result = model.fit(y, params, x=x, nan_policy='omit')
    return result, model

#Params: numpy.ndarray x - x values
#        numpy.ndarray y - y values
#        int num_basis_gaussians - number of gaussians to fit
#        list amplitudes - List of the amplitudes of the gaussians
#        list centers - List of the centers of the gaussians
#        list sigmas - List of the std deviations of the gaussians
#Does: Function to fit gaussian derivatives using lmfit with positive constraints for amplitude and sigma
#Returns: Pair (lmfit.result, lmfit.model)- Pair of lmfit result and model
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


    result = model.fit(y, params, x=x, nan_policy='omit')
    return result, model
#Param: float ratio - percentage of gaussian peak height to use for a full width measurement to sigma
#Returns: The full width any max factor
#Does:Function for finding some width at some ratio of max from peak - similar to FWHM
def get_anymax_factor(ratio):
    if (ratio >= 1): #return FWHM if ratio is invalid
        print("full width any max has invalid ratio")
        return SMALL_FWHM_FACTOR
    else:
        return np.sqrt(8*np.log(1/ratio))
#Params: numpy.ndarray x - x values
#        numpy.ndarray y - y values
#        int peak_index - center of peak for the gaussian
#        float ratio - full width any max ratio eg. 1/2 1/3
#Returns: float - An estimation for the std deviation of the gaussian
#Does: Function to estimate sigmas of gaussians corresponding to the peak_index
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
    fwam = x[right_idx] - x[left_idx]
    sigma = abs(fwam / get_anymax_factor(ratio)) # Convert to sigma
    #cap sigma
    return min(sigma, MAX_SIGMA)
#Params: np array x - x values
#        np array y - y values
#        int peak_index - center of peak for the gaussian
#Returns: float - an average of different full width any max estimations
#Does: Calculates an average of different full width any max estimations with a range defined in constants.py
def estimate_average_sigma(x, y, peak_index):
    total = 0
    count = 0
    for i in range(ESTIMATE_SIGMA_ITERATIONS_START, ESTIMATE_SIGMA_ITERATIONS_END - 1):
        total += estimate_sigma(x,y,peak_index, i/ESTIMATE_SIGMA_ITERATIONS_END)
        count += 1
    return total/count

#Params: numpy.ndarray y - y values
#        int window_length - savgol filter window length
#        int polyorder - savgol filter poly order
#Returns: numpy.ndarray - if SMOOTHING is true in constants it returns a smoothed y. otherwise returns y
#Does: If smoothing is True in constants.py it will smooth y. Otherwise, it returns y.
def savgol_filter_bool(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER):
    if SMOOTHING:
        return savgol_filter(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    else:
        return y
#Params: numpy.ndarray y - y values
#        numpy.ndarray peaks - peak indices
#        dict peak_info - properties result from scip findPeaks
#Returns: Masked peak array
#Does: Filters peaks by relative and absolute height defined in constants.py
def filter_by_max_peak_height(y, peaks, peak_info):
    peaks_abs = abs(peak_info["peak_heights"]) > MIN_ABSOLUTE_PEAK_HEIGHT
    peaks_rel = abs(peak_prominences(y, peaks)[0]) > -MIN_PROMINENCE
    return peaks[peaks_abs & peaks_rel]

#Params: numpy.ndarray x - x values
#        numpy.ndarray y - y values
#        int num_gaussians- number of gaussians to make a guess for
#return tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray) - returns (peak centers, peak amplitudes, peak std deviations)
#Does: Function to generate initial guesses for Gaussian parameters
def generate_initial_guesses(x, y, num_gaussians):
    # Smooth the noisy data
    y_smoothed = savgol_filter_bool(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    # Calculate the numerical derivatives
    d_y_smoothed = np.gradient(y_smoothed,x)
    # Calculate the 2nd numerical derivatives
    dd_y = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = np.gradient(d_y_smoothed, x)
    dd_y_smoothed = savgol_filter_bool(dd_y_smoothed, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    # Find peaks in the negative second derivative (to locate the centers of Gaussians)
    prominence = PROMINENCE_PERECENT * np.nanmax(dd_y)
    height = HEIGHT_THRESHOLD * np.nanmax(dd_y)

    dd_y_peaks_all, peak_info = find_peaks(-dd_y_smoothed, height=height, distance=DISTANCE, prominence=prominence)

    #filter peaks
    dd_y_peaks_all = filter_by_max_peak_height(-dd_y_smoothed, dd_y_peaks_all, peak_info)
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
#Params: numpy.ndarray x - x values
#        numpy.ndarray y - y values
#        int num_basis - number of gaussians/gaussian derivatives to cut off at
#        float rss_threshold_percent- residual sum of squares threshold
#Returns: list (tuple (float, float | int)) - list of least impactful gaussian indices and rss contributions
#Does: Function to remove least impactful gaussians derivatives by contribution to RSS and cap at num_basis
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
            reduced_result = reduced_model.fit(y, reduced_params, x=x, nan_policy='omit')
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

#Params: numpy.ndarray x - x values
#        numpy.ndarray y - y values
#        lmfit.result result - result to remove gaussians from
#        float rss_threshold_percent- residual sum of squares threshold
#Returns list (tuple (float, float | int)) - returns a list of tuples representing the gaussian index and the rss increase
#Does: removes the least impactful gaussians from the result and returns a list of the result with them removed
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
            reduced_result = reduced_model.fit(y, reduced_params, x=x, nan_policy='omit')
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
#Params: x - x values
#        numpy.ndarray peaks - array of peak centers
#Returns: numpy.ndarray - numpy.ndarray of peaks after filter
#Does: Filters out peaks that have a separation of less MIN_PEAK_X_DISTANCE in constants.py
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

#TODO: documentation
def fit_gaussians_to_signal_reduced_result(x, z, reduced_result, CustomGaussianModel, VARY_CENTERS=True, PERCENTAGE_RANGE=10, THRESHOLD_PERCENT=RSS_THRESHOLD_PERCENT, PLOT = True):
    B_model = None
    B_params = lmfit.Parameters()

    remaining_indices = [i for i in range(len(reduced_result.params) // 3) if f'g{i}_center' in reduced_result.params]

    for i in remaining_indices:
        g = CustomGaussianModel(prefix=f'g{i}_')
        if B_model is None:
            B_model = g
        else:
            B_model = B_model + g
        B_params.update(g.make_params())

        center_value = reduced_result.params[f'g{i}_center'].value
        amplitude_value = reduced_result.params[f'g{i}_amplitude'].value
        sigma_value = reduced_result.params[f'g{i}_sigma'].value

        B_params.add(f'g{i}_center', value=center_value, vary=VARY_CENTERS)
        B_params.add(f'g{i}_amplitude', value=amplitude_value)
        B_params.add(f'g{i}_sigma', value=sigma_value)

    print(f"Fitting B-terms with initial guesses: {B_params}...")
    B_result = B_model.fit(z, B_params, x=x)

    impactful_gaussian_B = remove_least_impactful_gaussians_by_fit(
        x, z, B_result, len(remaining_indices), rss_threshold_percent=THRESHOLD_PERCENT)
    print(f"Least impactful Gaussians are: {impactful_gaussian_B}")

    impactful_gaussian_B_indices = [i for i, _ in impactful_gaussian_B]
    reduced_B_model = None
    reduced_B_params = lmfit.Parameters()
    remaining_B_indices = [i for i in remaining_indices if i not in impactful_gaussian_B_indices]

    print(f"rem ind {remaining_B_indices}")
    for i in remaining_B_indices:
        g = CustomGaussianModel(prefix=f'g{i}_')
        if reduced_B_model is None:
            reduced_B_model = g
        else:
            reduced_B_model = reduced_B_model + g

        B_center_value = B_result.params[f'g{i}_center'].value
        B_amplitude_value = B_result.params[f'g{i}_amplitude'].value
        B_sigma_value = B_result.params[f'g{i}_sigma'].value

        reduced_B_params.add(f'g{i}_center', value=B_center_value, vary=VARY_CENTERS,
                             min=B_center_value - (B_center_value * PERCENTAGE_RANGE / 100),
                             max=B_center_value + (B_center_value * PERCENTAGE_RANGE / 100))
        reduced_B_params.add(f'g{i}_amplitude', value=B_amplitude_value,
                             min=B_amplitude_value - (B_amplitude_value * PERCENTAGE_RANGE / 100),
                             max=B_amplitude_value + (B_amplitude_value * PERCENTAGE_RANGE / 100))
        reduced_B_params.add(f'g{i}_sigma', value=B_sigma_value,
                             min=B_sigma_value - (B_sigma_value * PERCENTAGE_RANGE / 100),
                             max=B_sigma_value + (B_sigma_value * PERCENTAGE_RANGE / 100))

    reduced_B_result = reduced_B_model.fit(z, reduced_B_params, x=x)
    if PLOT: dplt.plot_fit_with_residuals(x, z, reduced_B_result.best_fit, title="Reduced B-Term Fit with Residuals")

    return reduced_B_result
#TODO: redo docs
def iterate_and_fit_gaussians(x, y, z, mcd_df, max_basis_gaussians=MAX_BASIS_GAUSSIANS, num_guesses=NUM_GUESSES):
    avg_bic_values = []
    avg_delta_bic_values = []
    all_fits = []

    amplitudes, centers, sigmas = generate_initial_guesses(x, y, max_basis_gaussians)
    previous_bic = None
    lowest_bic = float('inf')
    lowest_bic_idx = -1

    for num_basis in range(1, len(centers) + 1):
        bic_list = []
        fits = []

        for guess in range(num_guesses):
            result, model = fit_gaussians(x, y, num_basis, amplitudes[:num_basis], centers[:num_basis], sigmas[:num_basis])
            bic = result.bic
            print(f'result.bic:{bic}')
            bic_list.append(bic)
            fits.append(result)

        avg_bic = np.mean(bic_list)
        avg_bic_values.append(avg_bic)

        if previous_bic is not None:
            delta_bic = avg_bic - previous_bic
            avg_delta_bic_values.append(delta_bic)
            print(f'num_basis: {num_basis} | avg BIC = {avg_bic} | delta BIC: {delta_bic}')

            if avg_bic < lowest_bic:
                lowest_bic = avg_bic
                lowest_bic_idx = num_basis

            if abs(delta_bic) < DELTA_BIC_THRESHOLD:
                print(f"Delta BIC ~ 0 at N = {num_basis}. Reporting fits at N-1 = {num_basis - 1}.")
                all_fits = fits
                break

        previous_bic = avg_bic

    if not all_fits:
        if lowest_bic_idx > 1:
            print(f"Returning the N = {lowest_bic_idx} basis functions fit where BIC was minimized.")
            num_basis = lowest_bic_idx
        else:
            num_basis = 1
        all_fits = fits

    dplt.plot_gaussian_iterations(x, y, all_fits, num_basis, lowest_bic_idx)

    impactful_gaussians = remove_least_impactful_gaussians_by_fit(x, y, all_fits[-1], num_basis)
    print(f"Least impactful Gaussians are: {impactful_gaussians}")
    impactful_gaussian_indices = [i for i, _ in impactful_gaussians]

    print(f"Re-fitting after removing Gaussians: {impactful_gaussian_indices}...")

    reduced_result = fit_gaussians_to_signal_reduced_result(
        x, y, all_fits[-1], CustomGaussianModel,
        VARY_CENTERS=VARY_CENTERS,
        PERCENTAGE_RANGE=PERCENTAGE_RANGE,
        THRESHOLD_PERCENT=THRESHOLD_PERCENT
    )

    dplt.plot_reduced_result(x, y, num_basis, reduced_result, impactful_gaussian_indices)
    dplt.plot_bic(avg_bic_values, avg_delta_bic_values)

    remaining_indices = [i for i in range(num_basis) if i not in impactful_gaussian_indices]

    A_model = None
    A_params = lmfit.Parameters()

    for i in remaining_indices:
        g = CustomGaussian_ddx_Model(prefix=f'g{i}_')
        if A_model is None:
            A_model = g
        else:
            A_model = A_model + g
        A_params.update(g.make_params())

        center_value = reduced_result.params[f'g{i}_center'].value
        amplitude_value = reduced_result.params[f'g{i}_amplitude'].value
        sigma_value = reduced_result.params[f'g{i}_sigma'].value

        A_params.add(f'g{i}_center', value=center_value, vary=VARY_CENTERS)
        A_params.add(f'g{i}_amplitude', value=amplitude_value)
        A_params.add(f'g{i}_sigma', value=sigma_value, max=MAX_SIGMA)

    A_result = A_model.fit(z, A_params, x=x, nan_policy='omit')
    dplt.plot_A_terms(x, z, A_result, remaining_indices)

    impactful_gaussian_derivatives = remove_least_impactful_gaussian_derivatives_by_fit(
        x, z, A_result, len(remaining_indices), rss_threshold_percent=THRESHOLD_PERCENT)
    print(f"Least impactful Gaussians are: {impactful_gaussian_derivatives}")
    impactful_gaussian_derivative_indices = [i for i, _ in impactful_gaussian_derivatives]

    print(f"Re-fitting after removing Gaussians: {impactful_gaussian_derivative_indices}...")
    reduced_A_result = fit_gaussians_to_signal_reduced_result(
        x, z, A_result, CustomGaussian_ddx_Model,
        VARY_CENTERS=VARY_CENTERS,
        PERCENTAGE_RANGE=PERCENTAGE_RANGE,
        THRESHOLD_PERCENT=THRESHOLD_PERCENT
    )

    dplt.plot_xz_after_gaussian_removal(
        x, z, reduced_A_result,
        [i for i in remaining_indices if i not in impactful_gaussian_derivative_indices],
        impactful_gaussian_derivative_indices
    )

    ##############################################################################################################################

    z_minus_A = z - reduced_A_result.best_fit

    mask_pos = z_minus_A > 0
    mask_neg = z_minus_A < 0

    x_pos = x[mask_pos]
    z_pos = z_minus_A[mask_pos]

    x_neg = x[mask_neg]
    z_neg = z_minus_A[mask_neg]

    reduced_B_result_pos = fit_gaussians_to_signal_reduced_result(
        x_pos, z_pos, reduced_result, CustomGaussianModel,
        VARY_CENTERS=VARY_CENTERS,
        PERCENTAGE_RANGE=PERCENTAGE_RANGE,
        THRESHOLD_PERCENT=THRESHOLD_PERCENT
    )

    reduced_B_result_neg = fit_gaussians_to_signal_reduced_result(
        x_neg, z_neg, reduced_result, CustomGaussianModel,
        VARY_CENTERS=VARY_CENTERS,
        PERCENTAGE_RANGE=PERCENTAGE_RANGE,
        THRESHOLD_PERCENT=THRESHOLD_PERCENT
    )

    # Allocate combined fit array
    combined_B_fit = np.zeros_like(z)

    # Use the **same masks** for assignment
    combined_B_fit[mask_pos] = reduced_B_result_pos.best_fit
    combined_B_fit[mask_neg] = reduced_B_result_neg.best_fit

    # Add A-terms back in to reconstruct full modeled signal
    final_fit = reduced_A_result.best_fit + combined_B_fit

    # Wrap in a combined result object for plotting and inspection
    class CombinedResult:
        def __init__(self, best_fit, A_result, B_result):
            self.best_fit = best_fit
            self.A_result = A_result
            self.B_result = B_result

    reduced_AB_result = CombinedResult(final_fit, reduced_A_result, reduced_B_result_pos)

    # Plot original z vs combined A+B model
    dplt.plot_final_model(x, z, reduced_AB_result.best_fit)

    ##############################################################################################################################

    # Report the parameters of interest.
    print("\n========= A-Term Parameters (Derivatives) =========")
    for name, param in reduced_A_result.params.items():
        if param.stderr is not None:
            print(f"{name}: {param.value:.5g} ± {param.stderr:.2g}")
        else:
            print(f"{name}: {param.value:.5g}")

    print("\n========= B-Term Parameters =========")
    if hasattr(reduced_B_result_pos, 'params'):
        for name, param in reduced_B_result_pos.params.items():
            if param.stderr is not None:
                print(f"{name}: {param.value:.5g} ± {param.stderr:.2g}")
            else:
                print(f"{name}: {param.value:.5g}")
    else:
        print("B-term result has no parameters.")

    print("\n========= Dipole Parameters (Raw fit before A/B separation) =========")
    for name, param in reduced_result.params.items():
        if param.stderr is not None:
            print(f"{name}: {param.value:.5g} ± {param.stderr:.2g}")
        else:
            print(f"{name}: {param.value:.5g}")

    # Extract parameters from A and dipole fits
    aterm_params = reduced_A_result.params
    dipole_params = reduced_result.params

    # Pass the x-values (wavenumbers) to calculate the range
    x_values = mcd_df['wavenumber'].values

    # Group centers and calculate A/D ratio
    grouped_data = group_centers(dipole_params, aterm_params, x_values, tolerance_percentage=TOLERANCE_X)

    # Create and display result table
    df = pd.DataFrame(grouped_data)
    print("\n========= Grouped A/D Fit Table =========")
    print(df)

    return df
