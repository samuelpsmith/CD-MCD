import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import re
from collections import defaultdict
import traceback

# Set up logging
logging.basicConfig(filename='mcd_processing.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Suppress matplotlib font logging
import matplotlib
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

def load_json(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            logging.info(f"Loaded data from {file_path}")
            return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error parsing JSON file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading JSON file: {e}")
        raise

def select_files() -> defaultdict:
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select CSV files for MCD, Absorption, and Sticks processing")
    file_dict = defaultdict(dict)
    
    if file_paths:
        for file_path in file_paths:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if re.search(r'pos', base_name, re.IGNORECASE):
                file_dict[base_name.replace('pos', '', 1).replace('POS', '', 1)]['pos'] = file_path
            elif re.search(r'neg', base_name, re.IGNORECASE):
                file_dict[base_name.replace('neg', '', 1).replace('NEG', '', 1)]['neg'] = file_path
            elif re.search(r'abs', base_name, re.IGNORECASE):
                file_dict[base_name.replace('abs', '', 1).replace('ABS', '', 1)]['abs'] = file_path
            elif re.search(r'sticks', base_name, re.IGNORECASE):
                file_dict[base_name.replace('sticks', '', 1).replace('STICKS', '', 1)]['sticks'] = file_path
            else:
                messagebox.showerror("File Naming Error", f"File {base_name} does not contain 'pos', 'neg', 'abs', or 'sticks'. Please rename the file accordingly.")

    return file_dict

def create_output_directory(base_path: str) -> str:
    output_dir = os.path.join(base_path, "processed_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def read_csv_file(filename: str, column_names: list = None) -> pd.DataFrame:
    try:
        if column_names:
            df = pd.read_csv(filename, names=column_names)
        else:
            df = pd.read_csv(filename)
        df = df.sort_values(by='wavelength').reset_index(drop=True)
        logging.info(f"Read and sorted file {filename} successfully")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
    except pd.errors.EmptyDataError:
        logging.error(f"Empty file: {filename}")
    except pd.errors.ParserError:
        logging.error(f"Parsing error in file: {filename}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    return None

def calculate_differences(positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> tuple:
    x_diff = (positive_df['x_pos'] - negative_df['x_neg']) / 2
    y_diff = (positive_df['y_pos'] - negative_df['y_neg']) / 2
    x_stdev = np.sqrt(2 * ((positive_df['std_dev_x'] ** 2) + (negative_df['std_dev_x'] ** 2)))
    y_stdev = np.sqrt(2 * ((positive_df['std_dev_y'] ** 2) + (negative_df['std_dev_y'] ** 2)))
    R = np.sqrt(x_diff ** 2 + y_diff ** 2)
    R_stdev = np.sqrt(((x_diff * x_stdev / R) ** 2) + ((y_diff * y_stdev / R) ** 2))
    R_signed = R * np.sign(y_diff)
    return x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev

def interpolate_data(wavelength: pd.Series, R_signed: pd.Series, spline_points: int) -> tuple:
    try:
        spline = make_interp_spline(wavelength, R_signed)
        X_ = np.linspace(wavelength.min(), wavelength.max(), spline_points)
        Y_ = spline(X_)
        return X_, Y_
    except ValueError as e:
        logging.error(f"Interpolation error: {e}")
        return wavelength, R_signed

def kk_arbspace(omega: np.ndarray, imchi: np.ndarray, alpha: int) -> np.ndarray:
    omega = np.array(omega) 
    imchi = np.array(imchi)

    if omega.ndim == 1:
        omega = omega[np.newaxis, :]
    if imchi.ndim == 1:
        imchi = imchi[np.newaxis, :]

    g = omega.shape[1]
    rechi = np.zeros_like(imchi)
    a = np.zeros_like(imchi)
    b = np.zeros_like(imchi)
    deltaomega = omega[0, 1] - omega[0, 0]

    for j in range(g):
        alpha1, beta1 = 0, 0
        if j > 0:
            for k in range(j):
                a[0, j] = (alpha1 + (omega[0, k + 1] - omega[0, k]) * (imchi[0, k] * omega[0, k] ** (2 * alpha + 1) / (omega[0, k] ** 2 - omega[0, j] ** 2)))
                alpha1 = a[0, j]
        for k in range(j + 1, g):
            b[0, j] = (beta1 + (omega[0, k] - omega[0, k-1]) * (imchi[0, k] * omega[0, k] ** (2 * alpha + 1) / (omega[0, k] ** 2 - omega[0, j] ** 2)))
            beta1 = b[0, j]
        rechi[0, j] = 2 / np.pi * (a[0, j] + b[0, j]) * omega[0, j] ** (-2 * alpha)
    
    return rechi.flatten()

def convert_abs_to_extinction(df: pd.DataFrame, filename: str, abs_data: dict, columns: list) -> pd.DataFrame:
    if filename in abs_data:
        concentration = abs_data[filename]['concentration_mol_L']
        pathlength = abs_data[filename]['pathlength_cm']
        for column in columns:
            df[f'{column}_extinction'] = df[column] / (concentration * pathlength)
        logging.info(f"Converted {columns} to extinction for {filename}")
    else:
        logging.warning(f"No absorbance data found for {filename}, conversion skipped")
    return df

def scale_sticks(sticks_df: pd.DataFrame, max_absorbance: float, scale_factor: float = 1) -> pd.DataFrame:
    sticks_df = sticks_df.copy()
    sticks_df['scaled_strength'] = sticks_df['strength'] / sticks_df['strength'].max() * max_absorbance * scale_factor
    return sticks_df

def gaussian_derivative_wavenumber(k, amplitude, mean, stddev):
    #remove the negative sign before amplitude to account for the change in sign when converting to wavenumber.
    return amplitude * (k - mean) / (stddev ** 2) * np.exp(-((k - mean) ** 2) / (2 * stddev ** 2))

def pick_peaks(df, column='intensity_extinction', height_percent=1):
    """Picks the peaks of the absorption data.

    Args:
        df (pd.DataFrame): The DataFrame containing the absorption data.
        column (str, optional): The column to pick peaks from. Defaults to 'intensity_extinction'.
        height_percent (float, optional): The minimum height of a peak as a percentage of the maximum value.
        
    Returns:
        tuple: Arrays of peak indices, peak wavelengths and peak intensities.
    """

    print('starting pick_peaks')
    height = df[column].max() * (height_percent / 100)
    peaks, _ = find_peaks(df[column], height=height, prominence=0.25 * height)
    print("finished find peaks call")
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])
    peak_wavelengths = df.iloc[peaks]['wavelength'].values
    peak_intensities = df.iloc[peaks][column].values
    print(f"Peak wavelengths: {peak_wavelengths}")
    return peaks, peak_wavelengths, peak_intensities

def find_local_maxima_within_range(x_data, y_data, center, sigma):
    lower_bound = max(center - 3 * sigma, x_data.min())
    upper_bound = min(center + 3 * sigma, x_data.max())
    
    fitting_range = (x_data >= lower_bound) & (x_data <= upper_bound)
    print(f"fitting_range: {fitting_range}, lower_bound: {lower_bound}, upper_bound: {upper_bound}")
    
    if np.any(fitting_range):
        local_max_index = np.argmax(y_data[fitting_range])
        local_max_x = x_data[fitting_range][local_max_index]
        local_max_y = y_data[fitting_range][local_max_index]
        return local_max_x, local_max_y
    return center, 0

def fit_peak(x_data, y_data, initial_guess):
    """Fit the peak using the gaussian_derivative_wavenumber function.
    
    Args:
        x_data (np.ndarray): The x-axis data (wavenumber).
        y_data (np.ndarray): The y-axis data.
        initial_guess (list): The initial guess for the fitting parameters.
        
    Returns:
        tuple: The fitted parameters and the R² value.
    """
    try:
        popt, pcov = curve_fit(gaussian_derivative_wavenumber, x_data, y_data, p0=initial_guess)
        residuals = y_data - gaussian_derivative_wavenumber(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, r_squared
    except Exception as e:
        print(f"An error occurred during fitting for initial guess {initial_guess}: {e}")
        try:
            # Try fitting with the opposite sign of the amplitude
            initial_guess[0] = -initial_guess[0]
            popt, pcov = curve_fit(gaussian_derivative_wavenumber, x_data, y_data, p0=initial_guess)
            residuals = y_data - gaussian_derivative_wavenumber(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            return popt, r_squared
        except Exception as e:
            print(f"An error occurred during fitting with opposite sign for initial guess {initial_guess}: {e}")
            return None, None

def fit_peaks_separately(abs_df, mcd_df, column='intensity_extinction', height_percent=1):
    """Fit peaks separately using the gaussian_derivative_wavenumber function.
    
    Args:
        abs_df (pd.DataFrame): The DataFrame containing the absorption data.
        mcd_df (pd.DataFrame): The DataFrame containing the MCD data.
        column (str, optional): The column to pick peaks from. Defaults to 'intensity_extinction'.
        height_percent (float, optional): The minimum height of a peak as a percentage of the maximum value.
        
    Returns:
        list: Fitted parameters for each peak.
    """
    print("Starting peak fitting process...")

    # Step 1: Pick peaks from the absorption data
    peaks, peak_wavelengths, peak_intensities = pick_peaks(abs_df, column=column, height_percent=height_percent)
    print(f"Picked peaks: {peaks}")
    print(f"Peak wavelengths: {peak_wavelengths}")
    print(f"Peak intensities: {peak_intensities}")

    if len(peaks) == 0:
        print("no peaks found")
        return []

    # Step 2: Convert wavelength to wavenumber in the MCD data
    mcd_df['wavenumber'] = 1e7 / mcd_df['wavelength']
    print("Converted wavelength to wavenumber in MCD data.")

    # Step 3: Normalize MCD intensity by wavenumber
    mcd_df['normalized_intensity'] = mcd_df['R_signed_extinction'] / mcd_df['wavenumber']
    print("Normalized MCD intensity by wavenumber.")

    # Plot the normalized intensity vs. wavenumber for MCD data
    plt.figure(figsize=(8, 6))
    plt.plot(mcd_df['wavenumber'], mcd_df['normalized_intensity'], label='Normalized Intensity')
    plt.xlabel('Wavenumber')
    plt.ylabel('Normalized Intensity')
    plt.title('Normalized Intensity vs. Wavenumber for MCD Data')
    plt.legend()
    plt.show()

    fitted_params = []
    for peak_index in peaks:
        try:
            peak_wavelength = abs_df.iloc[peak_index]['wavelength']
            sigma_wavelength = (peak_widths(abs_df[column], [peak_index], rel_height=0.5)[0][0]) / (2 * np.sqrt(2 * np.log(2)))
            peak_wavenumber = 1e7 / peak_wavelength
            #sigma_min_wavenumber = 1e7 / (peak_wavelength + 3 * sigma_wavelength)
            #sigma_max_wavenumber = 1e7 / (peak_wavelength - 3 * sigma_wavelength)
            #sigma_wavenumber = (sigma_max_wavenumber - sigma_min_wavenumber) 
            #print("sigma_wavenumber:", sigma_wavenumber)

            sigma_max_wavenumber = 1e7 / (peak_wavelength + 3 * sigma_wavelength)
            sigma_wavenumber = (sigma_max_wavenumber - peak_wavenumber) 
            print("sigma_wavenumber:", sigma_wavenumber)

            print(f"Processing peak at index {peak_index}: wavelength={peak_wavelength}, wavenumber={peak_wavenumber}, sigma_wavelength={sigma_wavelength}, sigma_wavenumber={sigma_wavenumber}")

            # Step 4: Find local maxima for MCD within the range
            local_max_x, local_max_y = find_local_maxima_within_range(mcd_df['wavenumber'], mcd_df['normalized_intensity'], peak_wavenumber, sigma_wavenumber)
            print(f"Found local maxima: x={local_max_x}, y={local_max_y}")

            # Step 5: Fit the peak using MCD data
            initial_guess = [local_max_y, peak_wavenumber, sigma_wavenumber]
            print('initial guess:', initial_guess)
            popt, r_squared = fit_peak(mcd_df['wavenumber'], mcd_df['normalized_intensity'], initial_guess)
            print(f"Fitted parameters: {popt}, R²: {r_squared}")

            if popt is not None:
                fitted_params.append((popt, r_squared))
                print()

        except Exception as e:
            print(f"Error fitting peak at index {peak_index}: {e}")
            traceback.print_exc()
            continue
    
    print("Finished peak fitting process.")

    # Plot the fitted lineshapes
    plt.figure(figsize=(8, 6))
    plt.plot(mcd_df['wavenumber'], mcd_df['normalized_intensity'], label='Normalized Intensity')
    for params, r_squared in fitted_params:
        if params is not None:
            amplitude, mean, stddev = params
            fitted_curve = gaussian_derivative_wavenumber(mcd_df['wavenumber'], amplitude, mean, stddev)
            plt.plot(mcd_df['wavenumber'], fitted_curve, label=f'Fit (R² = {r_squared:.2f})')
    plt.xlabel('Wavenumber')
    plt.ylabel('Normalized Intensity')
    plt.title('Normalized Intensity vs. Wavenumber with Fitted Lineshapes')
    plt.legend()
    plt.show()

    return fitted_params

def plot_data(base_name, mcd_df: pd.DataFrame, abs_df: pd.DataFrame, mord_df: pd.DataFrame, config: dict, output_file_path: str, fitted_params, sticks_df: pd.DataFrame = None):
    abs_df['wavelength'] = pd.to_numeric(abs_df['wavelength'], errors='coerce')
    mord_df['wavelength'] = pd.to_numeric(mord_df['wavelength'], errors='coerce')
    if sticks_df is not None:
        sticks_df['wavelength'] = pd.to_numeric(sticks_df['wavelength'], errors='coerce')

    abs_df = abs_df[(abs_df['wavelength'] >= mcd_df['wavelength'].min()) & (abs_df['wavelength'] <= mcd_df['wavelength'].max())]
    mord_df = mord_df[(mord_df['wavelength'] >= mcd_df['wavelength'].min()) & (mord_df['wavelength'] <= mcd_df['wavelength'].max())]

    mcd_df = mcd_df.dropna(subset=['wavelength', 'R_signed_extinction']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(mcd_df) == 0:
        logging.error("No valid MCD data")
        return

    mcd_spec = mcd_df['R_signed_extinction']
    mcd_boxcar = mcd_spec.rolling(window=config['window_size'], center=True).mean().fillna(mcd_spec)
    mcd_X_, mcd_Y_ = interpolate_data(mcd_df['wavelength'], mcd_boxcar, config['spline_points'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 18))
    fig.subplots_adjust(hspace=0)

    abs_df = abs_df.dropna(subset=['wavelength', 'intensity_extinction']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(abs_df) == 0:
        logging.error("No valid Absorption data")
        return
    
    abs_spec = abs_df['intensity_extinction']
    abs_boxcar = abs_spec.rolling(window=config['window_size'], center=True).mean().fillna(abs_spec)
    abs_X_, abs_Y_ = interpolate_data(abs_df['wavelength'], abs_boxcar, config['spline_points'])
    
    ax1.plot(abs_df['wavelength'], abs_spec, label='Extinction', marker=' ', linestyle='-', color='blue')
    if config['plot_rolling_avg']:
        ax1.plot(abs_df['wavelength'], abs_boxcar, label='Extinction Boxcar Averaged', marker='x', linestyle='--', color='red')
    if config['plot_spline']:
        ax1.plot(abs_X_, abs_Y_, label='Extinction Spline Interpolation', linestyle='-', color='green')
    
    ax1.set_ylabel(r'Molar Extinction, $\epsilon$')
    ax1.grid(False)
    ax1.axhline(y=0, color='dimgrey', linestyle='-')
    ax1.set_title(f'{base_name[:-1]}')

    peaks, peak_wavelengths, peak_intensities = pick_peaks(abs_df, column= 'intensity_extinction', height_percent = 1)
    ax1.plot(peak_wavelengths, peak_intensities, 'ro')
    for wavelength, intensity in zip(peak_wavelengths, peak_intensities):
        ax1.text(wavelength, intensity, f'{wavelength:.1f}', ha='right', va='bottom', fontsize=10, color='red')

    if sticks_df is not None:
        sticks_df = sticks_df[(sticks_df['wavelength'] >= abs_df['wavelength'].min()) & (sticks_df['wavelength'] <= abs_df['wavelength'].max())]
        max_absorbance = abs_spec.max()
        sticks_df = scale_sticks(sticks_df, max_absorbance)
        ax1_sticks = ax1.twinx()
        
        for _, row in sticks_df.iterrows():
            ax1_sticks.plot([row['wavelength'], row['wavelength']], [0, row['scaled_strength']], linestyle='-', color='black')
        
        primary_ylim = ax1.get_ylim()
        ax1_sticks.set_ylim(primary_ylim)
        ax1_sticks.set_ylabel('Scaled Dipole Strength, Extinction')

    if config['plot_original']:
        ax2.plot(mcd_df['wavelength'], mcd_spec, label=r'Measured MCD $\Delta \epsilon / T$', marker=' ', linestyle='-', color='blue')
    if config['plot_rolling_avg']:
        ax2.plot(mcd_df['wavelength'], mcd_boxcar, label='Rolling Averaged R_signed_extinction', marker='x', linestyle='--', color='red')
    if config['plot_spline']:
        ax2.plot(mcd_X_, mcd_Y_, label='Spline Interpolation R_signed_extinction', linestyle='-', color='green')
    if config['plot_error_bars']:
        ax2.errorbar(mcd_df['wavelength'], mcd_spec, yerr=mcd_df['std_dev_extinction'], fmt='o', label='Error bars', ecolor='gray', alpha=0.5)
    ax2.set_ylabel(r'MCD, $\Delta \epsilon / T$')
    ax2.grid(False)
    ax2.axhline(y=0, color='dimgrey', linestyle='-')

    mord_df = mord_df.dropna(subset=['wavelength', 'mord']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(mord_df) == 0:
        logging.error("No valid MORD data")
        return

    mord_spec = mord_df['mord']
    mord_boxcar = mord_spec.rolling(window=config['window_size'], center=True).mean().fillna(mord_spec)
    mord_X_, mord_Y_ = interpolate_data(mord_df['wavelength'], mord_boxcar, config['spline_points'])
    
    if config['plot_original']:
        ax3.plot(mord_df['wavelength'], mord_spec, label='Measured MORD', marker=' ', linestyle='-', color='blue')
    if config['plot_rolling_avg']:
        ax3.plot(mord_df['wavelength'], mord_boxcar, label='Rolling Averaged MORD', marker='x', linestyle='--', color='red')
    if config['plot_spline']:
        ax3.plot(mord_X_, mord_Y_, label='Spline Interpolation MORD', linestyle='-', color='green')
    if config['plot_error_bars']:
        ax3.errorbar(mord_df['wavelength'], mord_spec, yerr=mord_df['std_dev'], fmt='o', label='Error bars', ecolor='gray', alpha=0.5)
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel(r'MORD, $\Theta / M * m * T$')
    ax3.grid(False)
    ax3.axhline(y=0, color='dimgrey', linestyle='-')

    if config['plot_derivative']:
        abs_df['intensity_derivative'] = np.gradient(abs_df['intensity'], abs_df['wavelength'])
        abs_df['intensity_derivative'] *= mcd_spec.max() / abs_df['intensity_derivative'].max()
        ax2.plot(abs_df['wavelength'], abs_df['intensity_derivative'], label='1st Derivative of Abs.', linestyle='--', color='purple')
        ax2.legend()

        abs_df['intensity_2nd_derivative'] = np.gradient(abs_df['intensity_derivative'], abs_df['wavelength'])
        abs_df['intensity_2nd_derivative'] *= mord_spec.max() / abs_df['intensity_2nd_derivative'].max()
        ax3.plot(abs_df['wavelength'], abs_df['intensity_2nd_derivative'], label='2nd Derivative of Abs.', linestyle='--', color='orange')
        ax3.legend()
    
    for params in fitted_params:
        amplitude, mean, stddev = params
        ax1.plot(1e7 / mean, amplitude, 'ro')
        ax1.text(1e7 / mean, amplitude, f'{1e7 / mean:.2f}', fontsize=12, color='red')

    output_plot_path = os.path.join(os.path.dirname(output_file_path), base_name + '_plot.png')
    plt.savefig(output_plot_path)
    plt.show()
    plt.close(fig)

def save_data(output_file_path, mcd_df, abs_df_copy, R_signed_averaged_filled, sticks_df=None):
    merged_df = pd.merge(mcd_df, abs_df_copy[['wavelength', 'intensity_extinction']], on='wavelength', how='inner')
    if sticks_df is not None:
        merged_df = pd.merge(merged_df, sticks_df[['wavelength', 'strength']], on='wavelength', how='left')
        merged_df.rename(columns={'strength': 'sticks'}, inplace=True)
    else:
        merged_df['sticks'] = np.nan

    merged_df['R_signed_averaged_filled'] = R_signed_averaged_filled
    merged_df.to_csv(output_file_path, index=False)
    logging.info(f"Data saved to {output_file_path}")

def process_files(file_dict: defaultdict, config: dict, abs_data: dict):
    column_names_pos = ['wavelength', 'x_pos', 'y_pos', 'R', 'theta', 'std_dev_x', 'std_dev_y', 'additional']
    column_names_neg = ['wavelength', 'x_neg', 'y_neg', 'R', 'theta', 'std_dev_x', 'std_dev_y', 'additional']
    column_names_abs = ['wavelength', 'intensity']
    column_names_sticks = ['wavelength', 'strength']

    for base_name, files in file_dict.items():
        if 'pos' in files and 'neg' in files and 'abs' in files:
            pos_file = files['pos']
            neg_file = files['neg']
            abs_file = files['abs']
            sticks_file = files.get('sticks', None)
            logging.info(f"Processing files: {pos_file}, {neg_file}, {abs_file}, and {sticks_file}")

            positive_df = read_csv_file(pos_file, column_names_pos)
            negative_df = read_csv_file(neg_file, column_names_neg)
            abs_df = read_csv_file(abs_file, column_names_abs)
            sticks_df = read_csv_file(sticks_file, column_names_sticks) if sticks_file else None

            if positive_df is not None and negative_df is not None and abs_df is not None:
                try:
                    abs_df_copy = abs_df.copy()

                    if config['convert_to_extinction']:
                        abs_df_copy = convert_abs_to_extinction(abs_df_copy, os.path.basename(abs_file), abs_data, ['intensity'])
                    
                    x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev = calculate_differences(positive_df, negative_df)
                    
                    mcd_df = pd.DataFrame({
                        'wavelength': positive_df['wavelength'],
                        'R_signed': R_signed,
                        'std_dev': R_stdev
                    })

                    if config['convert_to_extinction']:
                        mcd_df = convert_abs_to_extinction(mcd_df, os.path.basename(abs_file), abs_data, ['R_signed', 'std_dev'])

                    R_signed_averaged_filled = mcd_df['R_signed_extinction'].rolling(window=3, center=True).mean().fillna(mcd_df['R_signed_extinction'])

                    wavenumber_cm1 = 1e7 / mcd_df['wavelength'].values
                    mord_spectrum = kk_arbspace(wavenumber_cm1, mcd_df['R_signed_extinction'].values, alpha=0)
                    
                    mord_df = pd.DataFrame({
                        'wavelength': mcd_df['wavelength'],
                        'mord': mord_spectrum,
                        'std_dev': mcd_df['std_dev_extinction']
                    })

                    base_path = os.path.dirname(pos_file)
                    output_dir = create_output_directory(base_path)
                    output_file_path = os.path.join(output_dir, base_name + '_processed.csv')

                    if sticks_df is not None:
                        max_absorbance = abs_df_copy['intensity_extinction'].max()
                        sticks_df = scale_sticks(sticks_df, max_absorbance)
                        sticks_column = sticks_df['scaled_strength']
                    else:
                        sticks_column = pd.Series([None] * len(mcd_df['wavelength']), index=mcd_df.index)

                    fitted_params = fit_peaks_separately(abs_df_copy, mcd_df, column='intensity_extinction', height_percent=1)
                    plot_data(base_name, mcd_df, abs_df_copy, mord_df, config, output_file_path, fitted_params, sticks_df)

                    save_data(output_file_path, mcd_df, abs_df_copy, R_signed_averaged_filled, sticks_df)

                except Exception as e:
                    logging.error(f"Error processing files {pos_file}, {neg_file}, {abs_file}, and {sticks_file}: {e}")
                    messagebox.showerror("Processing Error", f"An error occurred: {e}")
            else:
                logging.error(f"One or more DataFrames for files {pos_file}, {neg_file}, {abs_file} are None")
        else:
            missing_types = [ftype for ftype in ['pos', 'neg', 'abs'] if ftype not in files]
            logging.error(f"Missing {', '.join(missing_types)} file(s) for base name {base_name}")
            messagebox.showerror("File Pairing Error", f"Missing {', '.join(missing_types)} file(s) for base name {base_name}")

def main():
    try:
        config = load_json('config.json')
        abs_data = load_json('abs_data.json')
        file_dict = select_files()
        if file_dict:
            process_files(file_dict, config, abs_data)
        else:
            logging.error("No files were selected or tagged properly.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    main()
