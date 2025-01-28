#
#
#cleans and interpolates data
#
#
import MCD_process.base.utils.logger as logger
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, interp1d
from scipy.signal import find_peaks

logging = logger.get_logger(__name__)

def calculate_differences(
    positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> tuple:
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

