import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import re
from collections import defaultdict

# Set up logging
logging.basicConfig(filename='mcd_processing.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Suppress matplotlib font logging
import matplotlib
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

def load_json(file_path: str) -> dict:
    """Loads data from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The data loaded from the JSON file.
    """
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
    """Opens a file dialog to select multiple files and tags each as positive, negative, absorption, or sticks.
    
    Returns:
        defaultdict: A dictionary of selected file paths tagged by type.
    """
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select CSV files for MCD, Absorption, and Sticks processing")
    file_dict = defaultdict(dict)
    
    if file_paths:
        for file_path in file_paths:
            base_name = os.path.splitext(os.path.basename(file_path))[0]  # This will strip the extension
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
    """Creates an output directory within the base path.
    
    Args:
        base_path (str): The base path where the output directory will be created.
        
    Returns:
        str: The path to the created output directory.
    """

    output_dir = os.path.join(base_path, "processed_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def read_csv_file(filename: str, column_names: list = None) -> pd.DataFrame:
    """Reads a CSV file into a DataFrame with specified column names if provided.
    
    Args:
        filename (str): The path to the CSV file.
        column_names (list, optional): The list of column names to use. Defaults to None.
        
    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    try:
        if column_names:
            df = pd.read_csv(filename, names=column_names)
        else:
            df = pd.read_csv(filename)
        df = df.sort_values(by='wavelength').reset_index(drop=True)  # Ensure the data is sorted by wavelength
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
    """Calculates differences and standard deviations between positive and negative field data.
    
    Args:
        positive_df (pd.DataFrame): The DataFrame with positive field data.
        negative_df (pd.DataFrame): The DataFrame with negative field data.
        
    Returns:
        tuple: A tuple containing x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev.
    """
    x_diff = (positive_df['x_pos'] - negative_df['x_neg']) / 2
    y_diff = (positive_df['y_pos'] - negative_df['y_neg']) / 2
    x_stdev = np.sqrt(2 * ((positive_df['std_dev_x'] ** 2) + (negative_df['std_dev_x'] ** 2)))
    y_stdev = np.sqrt(2 * ((positive_df['std_dev_y'] ** 2) + (negative_df['std_dev_y'] ** 2)))
    R = np.sqrt(x_diff ** 2 + y_diff ** 2)
    R_stdev = np.sqrt(((x_diff * x_stdev / R) ** 2) + ((y_diff * y_stdev / R) ** 2))
    R_signed = R * np.sign(y_diff)
    return x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev

def interpolate_data(wavelength: pd.Series, R_signed: pd.Series, spline_points: int) -> tuple:
    """Interpolates data for smoother plotting.
    
    Args:
        wavelength (pd.Series): The wavelength data.
        R_signed (pd.Series): The R_signed data.
        spline_points (int): The number of spline points for interpolation.
        
    Returns:
        tuple: A tuple containing interpolated X and Y data.
    """
    try:
        spline = make_interp_spline(wavelength, R_signed)
        X_ = np.linspace(wavelength.min(), wavelength.max(), spline_points)
        Y_ = spline(X_)
        return X_, Y_
    except ValueError as e:
        logging.error(f"Interpolation error: {e}")
        return wavelength, R_signed

def kk_arbspace(omega: np.ndarray, imchi: np.ndarray, alpha: int) -> np.ndarray:
    """Calculates the real part of susceptibility using Kramers-Kronig relations.
    
    Args:
        omega (np.ndarray): The frequency components.
        imchi (np.ndarray): The MCD response (extinction).
        alpha (int): The order of the spectral moment. Zero for zero-order analysis.
        
    Returns:
        np.ndarray: The real part of the susceptibility.
    
    Calculates the real part of susceptibility using Kramers-Kronig relations. 
    For MCD signal proccessing, omega is the vector of frequency components, imchi is the vector of MCD response (extinction). 
    Alpha is zero for zero-order analysis. Transforms of spectral moments higher than zero can be analyzed.
    In the case of magneto-optical activity, the complex function is the magnetically perturbed dielectric response.
    The real component of this complex function is MORD, the imaginary component the MCD.
    """

    #Convert omega and imchi to numpy arrays. 
    omega = np.array(omega) 
    imchi = np.array(imchi)

    #Ensure omega and imchi are row vectors. 
    if omega.ndim == 1:
        omega = omega[np.newaxis, :]
    if imchi.ndim == 1:
        imchi = imchi[np.newaxis, :]

    # Determine the size of the vectors, create and initialize rechi (the output vector) and a & b (the intermediate vectors) to zeros. 
    g = omega.shape[1] # g is the number of frequency points, the size of the vector. 
    rechi = np.zeros_like(imchi)
    a = np.zeros_like(imchi)
    b = np.zeros_like(imchi)
    deltaomega = omega[0, 1] - omega[0, 0] #delta omega is the frequnecy interval between consecutive points assuming uniform spacing. 

    #The discrete kramers-kronig analysis. 
    for j in range(g):
        alpha1, beta1 = 0, 0

        #for j > 0 calculate a for the jth frequency using the previous frequency points k. 
        if j > 0:
            for k in range(j):
                a[0, j] = (alpha1 + 
                           (omega[0, k + 1] - omega[0, k]) # this term represents the frequency interval
                           * (imchi[0, k] * omega[0, k] ** (2 * alpha + 1) / (omega[0, k] ** 2 - omega[0, j] ** 2))) # the Kramers-Kronig integrand. 
                alpha1 = a[0, j] # accumulates the sum for a 

        # for j + 1 to g calculate b using the subsequent frequency points k. 
        for k in range(j + 1, g):
            b[0, j] = (beta1 + (omega[0, k] - omega[0, k-1]) # this term represents the frequency interval. 
                       * (imchi[0, k] * omega[0, k] ** (2 * alpha + 1) / (omega[0, k] ** 2 - omega[0, j] ** 2))) #the Kramers-Kronig integrand. 
            beta1 = b[0, j] # accumulates the sum for b 

        # Calculate the real part of suceptibility 'rechi' at frequency 'j'    
        rechi[0, j] = 2 / np.pi * (a[0, j] + b[0, j]) * omega[0, j] ** (-2 * alpha)
    
    return rechi.flatten()

def convert_abs_to_extinction(df: pd.DataFrame, filename: str, abs_data: dict, columns: list) -> pd.DataFrame:
    """Converts absorbance to extinction using pathlength and concentration for specified columns.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        filename (str): The name of the file.
        abs_data (dict): The absorbance data dictionary.
        columns (list): The list of column names to convert.
        
    Returns:
        pd.DataFrame: The DataFrame with the converted extinction data.
    """

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
    """Scales the sticks' strengths to be less than the maximum value of the absorbance.
    
    Args:
        sticks_df (pd.DataFrame): The DataFrame containing the sticks data.
        max_absorbance (float): The maximum absorbance value to scale to.
        scale_factor (float, optional): The scaling factor. Defaults to 1.
        
    Returns:
        pd.DataFrame: The DataFrame with the scaled sticks data.
    """

    # Ensure sticks_df is a copy to avoid SettingWithCopyWarning
    sticks_df = sticks_df.copy()
    sticks_df['scaled_strength'] = sticks_df['strength'] / sticks_df['strength'].max() * max_absorbance * scale_factor
    return sticks_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

# Define the first derivative of the Gaussian function
def gaussian_derivative_wavenumber(k, amplitude, mean, stddev):
    return amplitude * (k - mean) / (stddev ** 2) * np.exp(-((k - mean) ** 2) / (2 * stddev ** 2))

def pick_peaks(df, column='abs', height_percent=1):
    """Picks the peaks of the absorption data.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the absorption data.
        column (str, optional): The column to pick peaks from. Defaults to 'abs'.
        height_percent (float, optional): The minimum height of a peak as a percentage of the maximum value.
        
    Returns:
        tuple: Arrays of peak indices, peak wavelengths and peak intensities.
    """
    height = df[column].max() * (height_percent / 100)
    peaks, _ = find_peaks(df[column], height=height, prominence=0.25 * height)
    peak_wavelengths = df.iloc[peaks]['wavelength'].values
    peak_intensities = df.iloc[peaks][column].values
    print(f"Peak wavelengths: {peak_wavelengths}")
    return peaks, peak_wavelengths, peak_intensities

def find_local_maxima_within_range(x_data, y_data, center, sigma):
    """Finds the local maximum of the y_data within 6 sigma range around the center.
    
    Args:
        x_data (np.ndarray): The x-axis data.
        y_data (np.ndarray): The y-axis data.
        center (float): The center value.
        sigma (float): The sigma value.
        
    Returns:
        tuple: The x and y values of the local maximum.
    """
    lower_bound = max(center - 3 * sigma, x_data.min())
    upper_bound = min(center + 3 * sigma, x_data.max())
    
    fitting_range = (x_data >= lower_bound) & (x_data <= upper_bound)
    
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
        np.ndarray: The fitted parameters.
    """
    try:
        popt, pcov = curve_fit(gaussian_derivative_wavenumber, x_data, y_data, p0=initial_guess)
        return popt
    except Exception as e:
        print(f"An error occurred during fitting for initial guess {initial_guess}: {e}")
        try:
            # Try fitting with the opposite sign of the amplitude
            initial_guess[0] = -initial_guess[0]
            popt, pcov = curve_fit(gaussian_derivative_wavenumber, x_data, y_data, p0=initial_guess)
            return popt
        except Exception as e:
            print(f"An error occurred during fitting with opposite sign for initial guess {initial_guess}: {e}")
            return None

def fit_peaks_separately(df, column='abs', height_percent=1):
    """Fit peaks separately using the gaussian_derivative_wavenumber function.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the absorption data.
        column (str, optional): The column to pick peaks from. Defaults to 'abs'.
        height_percent (float, optional): The minimum height of a peak as a percentage of the maximum value.
        
    Returns:
        list: Fitted parameters for each peak.
    """
    peaks, peak_wavelengths, peak_intensities = pick_peaks(df, column=column, height_percent=height_percent)
    
    # Convert wavelength to wavenumber
    df['wavenumber'] = 1e7 / df['wavelength']
    
    fitted_params = []
    for peak_index in peaks:
        peak_wavelength = df.iloc[peak_index]['wavelength']
        sigma = (peak_widths(df[column], [peak_index], rel_height=0.5)[0][0]) / (2 * np.sqrt(2 * np.log(2)))
        peak_wavenumber = 1e7 / peak_wavelength
        peak_intensity = df.iloc[peak_index][column]
        
        # Find local maxima for MCD
        local_max_x, local_max_y = find_local_maxima_within_range(df['wavenumber'], df[column], peak_wavenumber, sigma)
        
        initial_guess = [local_max_y, local_max_x, sigma]
        popt = fit_peak(df['wavenumber'], df[column], initial_guess)
        
        if popt is not None:
            fitted_params.append(popt)
    
    return fitted_params



def pick_peaks(df, column='intensity_extinction', height_percent=1):
    """Picks the peaks of the absorption data.

    
    Args:
        df (pd.DataFrame): The DataFrame containing the absorption data.
        column (str, optional): The column to pick peaks from. Defaults to 'intensity_extinction'.
        
    Returns:
        np.ndarray: Array of peak wavelengths.
    """

    height = ( df['intensity_extinction'].max() ) * (height_percent / 100)
    peaks, _ = find_peaks(df[column], height=height, prominence= 0.25*height)
    peak_wavelengths = df.iloc[peaks]['wavelength'].values
    peak_intensities = df.iloc[peaks]['intensity_extinction']
    print(f"Peak wavelengths: {peak_wavelengths}")
    return peak_wavelengths, peak_intensities


def plot_data(base_name, mcd_df: pd.DataFrame, abs_df: pd.DataFrame, mord_df: pd.DataFrame, config: dict, output_file_path: str, sticks_df: pd.DataFrame = None):
    """Plots the MCD, Absorption, and MORD data on a single figure with three stacked graphs.
    
    Optionally plots the derivatives of the absorption spectrum and sticks.
    
    Args:
        base_name (str): The base name of the file set.
        mcd_df (pd.DataFrame): The DataFrame with MCD data.
        abs_df (pd.DataFrame): The DataFrame with Absorption data.
        mord_df (pd.DataFrame): The DataFrame with MORD data.
        config (dict): The configuration dictionary.
        output_file_path (str): The path of the output file. 
        sticks_df (pd.DataFrame, optional): The DataFrame with sticks data. Defaults to None.
        
    """

    # Ensure 'wavelength' column is numeric
    abs_df['wavelength'] = pd.to_numeric(abs_df['wavelength'], errors='coerce')
    mord_df['wavelength'] = pd.to_numeric(mord_df['wavelength'], errors='coerce')
    if sticks_df is not None:
        sticks_df['wavelength'] = pd.to_numeric(sticks_df['wavelength'], errors='coerce')

    # Filter data based on the wavelength range in mcd_df
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

    # Plot Absorption data
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

    # Call the peak picking function
    peak_wavelengths, peak_intensities = pick_peaks(abs_df, column= 'intensity_extinction', height_percent = 1)

    # Plot and label the peaks
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
    
    output_plot_path = os.path.join(os.path.dirname(output_file_path), base_name + '_plot.png')
    plt.savefig(output_plot_path)

    plt.show()
    plt.close(fig)  # Close the figure to ensure it does not persist in the next iteration



def save_data(output_file_path, mcd_df, abs_df_copy, R_signed_averaged_filled, sticks_df=None):
    """Saves processed data to a CSV file.
    
    Args:
        output_file_path (str): The path to the output file.
        mcd_df (pd.DataFrame): The DataFrame with MCD data.
        abs_df_copy (pd.DataFrame): The DataFrame with Absorption data.
        R_signed_averaged_filled (pd.Series): The rolling averaged R_signed data.
        sticks_df (pd.DataFrame, optional): The DataFrame with sticks data. Defaults to None.
    """
    # Merge dataframes to align absorption data with MCD data
    merged_df = pd.merge(mcd_df, abs_df_copy[['wavelength', 'intensity_extinction']], on='wavelength', how='inner')

    # Add sticks data if available
    if sticks_df is not None:
        merged_df = pd.merge(merged_df, sticks_df[['wavelength', 'strength']], on='wavelength', how='left')
        merged_df.rename(columns={'strength': 'sticks'}, inplace=True)
    else:
        merged_df['sticks'] = np.nan

    merged_df['R_signed_averaged_filled'] = R_signed_averaged_filled

    merged_df.to_csv(output_file_path, index=False)
    logging.info(f"Data saved to {output_file_path}")


def process_files(file_dict: defaultdict, config: dict, abs_data: dict):
    """Processes each set of files and performs MCD, Absorption, and MORD analysis.
    
    Args:
        file_dict (defaultdict): The dictionary of selected files.
        config (dict): The configuration dictionary.
        abs_data (dict): The absorbance data dictionary.
    """
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
                        'std_dev': mcd_df['std_dev_extinction']  # Using converted std_dev for MORD
                    })

                    base_path = os.path.dirname(pos_file)
                    output_dir = create_output_directory(base_path)
                    output_file_path = os.path.join(output_dir, base_name + '_processed.csv')

                    # Call scale_sticks function if sticks_df is not None
                    if sticks_df is not None:
                        max_absorbance = abs_df_copy['intensity_extinction'].max()
                        sticks_df = scale_sticks(sticks_df, max_absorbance)
                        sticks_column = sticks_df['scaled_strength']
                    else:
                        sticks_column = pd.Series([None] * len(mcd_df['wavelength']), index=mcd_df.index)

                    plot_data(base_name, mcd_df, abs_df_copy, mord_df, config, output_file_path, sticks_df)

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
    """Main function to load configuration, select files, and process the data."""
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
