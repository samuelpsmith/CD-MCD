import os

import numpy
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from base import data_plotting as dplt
from base.data_fitting import iterate_and_fit_gaussians
from base.utils import file_handler as fh


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

    #plot the data frame
    dplt.plot_raw_df(mcd_df)

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


    # Optionally, display the table with improved formatting
        # df.style.format({
        #    'Dipole Center (D)': '{:.2f}',
        #    'A-Term Center (A)': '{:.2f}',
        #    'Amplitude (D)': '{:.4f}',
        #    'Amplitude (A)': '{:.4f}',
        #    'A/D Ratio': '{:.4f}'
        # })


# Main Execution (Reading from CSV)
if __name__ == "__main__":
    main()