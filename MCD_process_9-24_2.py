import os
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simpson
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import re
from collections import defaultdict

from plot_module import plot_data


logging.basicConfig(
    filename="mcd_processing.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)
print(f"Current working directory: {os.getcwd()}")


def load_json(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
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
    file_paths = filedialog.askopenfilenames(
        title="Select CSV files for MCD, Absorption, and Sticks processing"
    )
    file_dict = defaultdict(dict)

    if file_paths: # here I am trying to fix that case sensitivity problem.
        for file_path in file_paths:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if re.search(r"pos", base_name, re.IGNORECASE):
                file_dict[base_name.replace("pos", "", 1).replace("POS", "", 1)][
                    "pos"
                ] = file_path
            elif re.search(r"neg", base_name, re.IGNORECASE):
                file_dict[base_name.replace("neg", "", 1).replace("NEG", "", 1)][
                    "neg"
                ] = file_path
            elif re.search(r"abs", base_name, re.IGNORECASE):
                file_dict[base_name.replace("abs", "", 1).replace("ABS", "", 1)][
                    "abs"
                ] = file_path
            elif re.search(r"sticks", base_name, re.IGNORECASE):
                file_dict[base_name.replace("sticks", "", 1).replace("STICKS", "", 1)][
                    "sticks"
                ] = file_path
            else:
                messagebox.showerror(
                    "File Naming Error",
                    f"File {base_name} does not contain 'pos', 'neg', 'abs', or 'sticks'. Please rename the file accordingly.",
                )

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
        else: # I dont think this works, because later on I require column names to do basically any processing. 
            df = pd.read_csv(filename)
        df = df.sort_values(by="wavelength").reset_index(drop=True) #wonder what reset index does. 
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


def calculate_differences(
    positive_df: pd.DataFrame, negative_df: pd.DataFrame
) -> tuple:
    # This is a way to account for a baseline that is introduce by optical abberations of the setup, incl. linear dichroisms(?)
    # we do this parametrically ( by X and by Y)
    x_diff = (positive_df["x_pos"] - negative_df["x_neg"]) / 2
    y_diff = (positive_df["y_pos"] - negative_df["y_neg"]) / 2
    x_stdev = np.sqrt(
        2 * ((positive_df["std_dev_x"] ** 2) + (negative_df["std_dev_x"] ** 2))
    )
    # this is just a stats formula, but I better cite this. 
    y_stdev = np.sqrt(
        2 * ((positive_df["std_dev_y"] ** 2) + (negative_df["std_dev_y"] ** 2))
    )
    R = np.sqrt(x_diff**2 + y_diff**2) #pythagorus
    # cite all these. 
    R_stdev = np.sqrt(((x_diff * x_stdev / R) ** 2) + ((y_diff * y_stdev / R) ** 2))
    R_signed = R * np.sign(y_diff) # why do we multiply by the sign of y and not x? idk it I think its bc it looked better. figure it out. 
    return x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev


def interpolate_data(
    wavelength: pd.Series, R_signed: pd.Series, spline_points: int
) -> tuple:
    try:
        spline = make_interp_spline(wavelength, R_signed)
        X_ = np.linspace(wavelength.min(), wavelength.max(), spline_points)
        Y_ = spline(X_)
        return X_, Y_
    except ValueError as e:
        logging.error(f"Interpolation error: {e}")
        return wavelength, R_signed


def kk_arbspace(omega: np.ndarray, imchi: np.ndarray, alpha: int) -> np.ndarray:
    # kramers kronig function in arbitrary (frequency?) space
    # we ought to probably look this thing up and make sure we have it implemented correctly. 
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
                a[0, j] = alpha1 + (omega[0, k + 1] - omega[0, k]) * (
                    imchi[0, k]
                    * omega[0, k] ** (2 * alpha + 1)
                    / (omega[0, k] ** 2 - omega[0, j] ** 2)
                )
                alpha1 = a[0, j]
        for k in range(j + 1, g):
            b[0, j] = beta1 + (omega[0, k] - omega[0, k - 1]) * (
                imchi[0, k]
                * omega[0, k] ** (2 * alpha + 1)
                / (omega[0, k] ** 2 - omega[0, j] ** 2)
            )
            beta1 = b[0, j]
        rechi[0, j] = 2 / np.pi * (a[0, j] + b[0, j]) * omega[0, j] ** (-2 * alpha)

    return rechi.flatten()


def convert_abs_to_extinction(
    df: pd.DataFrame, filename: str, abs_data: dict, columns: list
) -> pd.DataFrame:
    print(
        f"starting convert to extinction. filename {filename}, abs_data {abs_data}, columns {columns}"
    )
    print(f"Filename: {filename}")
    print(f"Keys in abs_data: {abs_data.keys()}")

    if filename in abs_data:
        concentration = abs_data[filename]["concentration_mol_L"]
        pathlength = abs_data[filename]["pathlength_cm"]
        for column in columns:
            df[f"{column}_extinction"] = df[column] / (concentration * pathlength)
            #print("dataframe extinction", df[f"{column}_extinction"])
        logging.info(f"Converted {columns} to extinction for {filename}")
    else:
        logging.warning(f"No absorbance data found for {filename}, conversion skipped")
    return df


def scale_sticks(
    sticks_df: pd.DataFrame, max_absorbance: float, scale_factor: float = 1
) -> pd.DataFrame:
    sticks_df = sticks_df.copy()
    sticks_df["scaled_strength"] = (
        sticks_df["strength"]
        / sticks_df["strength"].max()
        * max_absorbance
        * scale_factor
    )
    return sticks_df


def pick_peaks(df, column="intensity_extinction", height_percent=1):
    """
    Picks the peaks of the absorption data.

    Args:
        df (pd.DataFrame): The DataFrame containing the absorption data.
        column (str, optional): The column to pick peaks from. Defaults to 'intensity_extinction'.
        height_percent (float, optional): The minimum height of a peak as a percentage of the maximum value.

    Returns:
        tuple: Arrays of peak indices, peak wavelengths, peak wavenumbers (if available), and peak intensities.
    """

    print("Starting pick_peaks")
    height = df[column].max() * (height_percent / 100)
    peaks, _ = find_peaks(df[column], height=height, prominence=0.25 * height)
    print("Finished find_peaks call")

    if len(peaks) == 0:
        # Return empty arrays or None for consistency
        peak_wavelengths = np.array([])
        peak_intensities = np.array([])
        peak_wavenumbers = np.array([]) if "wavenumber" in df.columns else None
    else:
        peak_wavelengths = df.iloc[peaks]["wavelength"].values
        peak_intensities = df.iloc[peaks][column].values
        if "wavenumber" in df.columns:
            peak_wavenumbers = df.iloc[peaks]["wavenumber"].values
            print(f"Peak wavenumbers: {peak_wavenumbers}")
        else:
            peak_wavenumbers = None
            print("Wavenumber column not found. Peak wavenumbers set to None.")

    # Return peaks, peak_wavelengths, peak_wavenumbers, peak_intensities
    return peaks, peak_wavelengths, peak_wavenumbers, peak_intensities


def find_local_maxima_within_range(
    x_data, y_data, center, gamma
):  # edited to use gamma
    lower_bound = max(
        center - ((3 * gamma) / 2), x_data.min()
    )  # is this in wavenumber or in indices?
    upper_bound = min(center + ((3 * gamma) / 2), x_data.max())

    fitting_range = (x_data >= lower_bound) & (x_data <= upper_bound)
    print(
        f"fitting_range: {fitting_range}, lower_bound: {lower_bound}, upper_bound: {upper_bound}"
    )

    if np.any(fitting_range):
        # Get the subset of x_data and y_data within the fitting range
        subset_x = x_data[fitting_range]
        subset_y = y_data[fitting_range]

        # Find the local maximum within this subset
        local_max_index = np.argmax(subset_y)
        local_max_x = subset_x.iloc[local_max_index]
        local_max_y = subset_y.iloc[local_max_index]
        return local_max_x, local_max_y

    print(
        "find_local_maxima_within_range unable to find maxima, no values in range. return center, 0"
    )
    return center, 0


def adjust_nearest_mean_points(df, means, column="normalized_intensity"):
    """
    Adjust the points nearest to the means in the DataFrame by averaging the nearest points.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        means (list): The list of mean values (wavenumbers) to find the nearest points to.
        column (str): The column to adjust. Defaults to 'normalized_intensity'.
    """
    for mean in means:
        nearest_idx = (df["wavenumber"] - mean).abs().idxmin()
        if nearest_idx + 1 < len(df) and nearest_idx - 1 >= 0:
            df.at[nearest_idx, column] = (
                df.at[nearest_idx - 1, column] + df.at[nearest_idx + 1, column]
            ) / 2
    return df


def index_to_xdata(xdata, indices):
    """Interpolate the values from signal.peak_widths to xdata."""
    ind = np.arange(len(xdata))
    f = interp1d(ind, xdata)
    return f(indices)


def plot_data_old(
    base_name: str,
    mcd_df: pd.DataFrame,
    abs_df: pd.DataFrame,
    mord_df: pd.DataFrame,
    config: dict,
    output_file_path: str,
    sticks_df: pd.DataFrame = None,
) -> None:
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
    mcd_X, mcd_Y = interpolate_data(
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
    abs_X, abs_Y = interpolate_data(
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
        peaks, peak_wavelengths, _, peak_intensities = pick_peaks(
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
        sticks_df = scale_sticks(sticks_df, max_absorbance)
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
    mord_X, mord_Y = interpolate_data(
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


def save_data(output_file_path, mcd_df, abs_df_copy, mord_df, sticks_df=None):
    # Merge MCD and Absorption data
    # considering renaming to mcd, abs, mord. But doing so would break some of the modularity of the code.

    print("sticks_df in savedata:", sticks_df) 
    merged_df = pd.merge(
        mcd_df.rename(columns={"R_signed_extinction": "mcd"}),
        abs_df_copy.rename(columns={"intensity_extinction": "abs"}),
        on="wavelength",
        how="inner",
    )

    # Merge in MORD data
    merged_df = pd.merge(
        merged_df,
        mord_df.rename(columns={"mord": "mord"}),
        on="wavelength",
        how="inner",
    )

    # Merge in sticks data, if available
    if sticks_df is not None:
        merged_df = pd.merge(
            merged_df,
            sticks_df[["wavelength", "strength"]].rename(
                columns={"strength": "sticks"}
            ),
            on="wavelength",
            how="outer",
        )
    else:
        merged_df["sticks"] = np.nan

    # Save the merged data to CSV with updated column names
    merged_df.to_csv(output_file_path, index=False)
    logging.info(f"Data saved to {output_file_path}")


def process_files(file_dict: defaultdict, config: dict, abs_data: dict):
    # here I am explicitly defining column names because the data comes unlabelled. 
    column_names_pos = [
        "wavelength",
        "x_pos",
        "y_pos",
        "R",
        "theta",
        "std_dev_x",
        "std_dev_y",
        "additional",
    ]
    column_names_neg = [
        "wavelength",
        "x_neg",
        "y_neg",
        "R",
        "theta",
        "std_dev_x",
        "std_dev_y",
        "additional",
    ]
    column_names_abs = ["wavelength", "intensity"]
    column_names_sticks = ["wavelength", "strength"]

    for base_name, files in file_dict.items():
        if "pos" in files and "neg" in files and "abs" in files:
            pos_file = files["pos"]
            neg_file = files["neg"]
            abs_file = files["abs"]
            sticks_file = files.get("sticks", None)
            logging.info(
                f"Processing files: {pos_file}, {neg_file}, {abs_file}, and {sticks_file}"
            )
            # read all those files in unless we dont have a sticks file then skip sticks
        
            positive_df = read_csv_file(pos_file, column_names_pos)
            negative_df = read_csv_file(neg_file, column_names_neg)
            abs_df = read_csv_file(abs_file, column_names_abs)
            sticks_df = (
                read_csv_file(sticks_file, column_names_sticks) if sticks_file else None
            )

            if (
                positive_df is not None
                and negative_df is not None
                and abs_df is not None
            ):
                try:
                    abs_df_copy = abs_df.copy()
                    #pass by ref vs pass by value.

                    # I dont know that we ever really want to look at this in units of abs? consider changing?

                    if config["convert_to_extinction"]:
                        print("convert to extinction is true (abs)")
                        abs_df_copy = convert_abs_to_extinction(
                            abs_df_copy,
                            os.path.basename(abs_file),
                            abs_data,
                            ["intensity"],
                        )

                    x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev = (
                        calculate_differences(positive_df, negative_df)
                    )

                    mcd_df = pd.DataFrame(
                        {
                            "wavelength": positive_df["wavelength"],
                            "R_signed": R_signed,
                            "std_dev": R_stdev,
                        }
                    )

                    if config["convert_to_extinction"]:
                        print("convert to extinction is true (mcd)")
                        mcd_df = convert_abs_to_extinction(
                            mcd_df,
                            os.path.basename(abs_file), # whats goin on here?
                            abs_data,
                            ["R_signed", "std_dev"],
                        )

                    R_signed_averaged_filled = (
                        mcd_df["R_signed_extinction"] # this is bad. This is a bad idea and we gotta not do this anymore. 
                        .rolling(window=3, center=True)
                        .mean()
                        .fillna(mcd_df["R_signed_extinction"])
                    ) # we DO need to fill in NaNs, but we ought not to rolling average data anymore. 

                    wavenumber_cm1 = 1e7 / mcd_df["wavelength"].values # gotta convert to wavenumber to do kramers kronig. 
                    mord_spectrum = kk_arbspace(
                        wavenumber_cm1, mcd_df["R_signed_extinction"].values, alpha=0
                    )  # should I used the fillna to handle null values before this?

                    mord_df = pd.DataFrame(
                        {
                            "wavelength": mcd_df["wavelength"],
                            "mord": mord_spectrum,
                            "std_dev": mcd_df["std_dev_extinction"],
                        }
                    )

                    base_path = os.path.dirname(pos_file)
                    output_dir = create_output_directory(base_path)
                    output_file_path = os.path.join(
                        output_dir, base_name + "processed.csv"
                    )
                    print("outputting to:", output_file_path)

                    if sticks_df is not None:
                        max_absorbance = abs_df_copy["intensity_extinction"].max()
                        sticks_df = scale_sticks(sticks_df, max_absorbance)
                        sticks_column = sticks_df["scaled_strength"]
                    else:
                        sticks_column = pd.Series(
                            [None] * len(mcd_df["wavelength"]), index=mcd_df.index
                        )

                    # fitting goes here. fitted_params = fit_peaks_seperately_old(abs_df_copy, mcd_df, column='intensity_extinction', height_percent=1, lorentz_frac=0.5)
                    plot_data(
                        base_name,
                        mcd_df,
                        abs_df_copy,
                        mord_df,
                        config,
                        output_file_path,
                        sticks_df,
                    )

                    print("sticks_df:", sticks_df)

                    save_data(
                        output_file_path,
                        mcd_df,
                        abs_df_copy,
                        mord_df,
                        sticks_df,
                    )

                except Exception as e:
                    logging.error(
                        f"Error processing files {pos_file}, {neg_file}, {abs_file}, and {sticks_file}: {e}"
                    )
                    messagebox.showerror("Processing Error", f"An error occurred: {e}")
            else:
                logging.error(
                    f"One or more DataFrames for files {pos_file}, {neg_file}, {abs_file} are None"
                )
        else:
            missing_types = [
                ftype for ftype in ["pos", "neg", "abs"] if ftype not in files
            ]
            logging.error(
                f"Missing {', '.join(missing_types)} file(s) for base name {base_name}"
            )
            messagebox.showerror(
                "File Pairing Error",
                f"Missing {', '.join(missing_types)} file(s) for base name {base_name}",
            )


def main():
    try:
        # Construct the absolute path to 'abs_data.json' based on the script's location
        # basically i iassume that abs_data.json, config.json and this script are all in the same directory.
        #i think I also had some issue where if you were running this on mac vs windows it didnt build the directory nmame right. (Case sensitvie?)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_data_path = os.path.join(script_dir, "abs_data.json")
        config_path = os.path.join(script_dir, "config.json")
        abs_data = load_json(abs_data_path)
        config = load_json(config_path)
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
