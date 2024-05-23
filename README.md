# CD-MCD
This project contains documentation for instrumentation at UNM's [Laboratory for Magneto-optical Spectroscopy](https://isco-op.unm.edu/centers/msl.html). Our current instrumentation includes broadband steady-state Circular Dichroism and Magnetic Circular Dichroism spectroscopy. This project also contains documentation for the data processing scripts used to interpret the data. 

## MCD Instrument. 

The Magnetic Circular Dichroism Spectrometer is a differential absorption spectrometer. We measure the difference in the absorption of left and right circularly polarized light. We can convert from MCD spectra to MORD spectra using the Kramers-Kronig relations. 

Read the wiki for a detailed description of how to use the MCD instrument. 
[Wiki Link](https://github.com/samuelpsmith/CD-MCD/wiki)

## MCD and Absorption Data Processing Script

This script is designed to process Magnetic Circular Dichroism (MCD) and absorption spectra data. It reads the data from CSV files, performs baseline correction and records standard deviations, converts absorbance to extinction, and plots the data. Additionally, it uses the Kramers-Kronig relations to calculate the MORD spectrum from the MCD spectrum.

### Features

- Reads and processes MCD and absorption data from CSV files.
- Converts absorbance to molar extinction coefficients.
- Calculates the real part of the susceptibility using the Kramers-Kronig relations.
- Plots MCD, Absorption, and MORD spectra on a single figure with three stacked graphs.
- Optionally plots the derivatives of the absorption spectrum and sticks for transition predictions.
- Saves the processed data to CSV files.

### Requirements

- Python 3.7 or higher
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - tkinter

You can install the required packages using the following command:

```bash
pip install pandas numpy matplotlib scipy tk
```

### Configuration Files

The script uses two configuration files:

1. `config.json`: General configuration for the script.
2. `abs_data.json`: Contains concentration and pathlength data for the absorption files.

#### Example `config.json`

```json
{
  "window_size": 3,
  "spline_points": 500,
  "convert_to_extinction": true,
  "plot_original": true,
  "plot_rolling_avg": true,
  "plot_spline": true,
  "plot_error_bars": true,
  "plot_derivative": true,
  "plot_sticks": true
}
```

#### Example `abs_data.json`

```json
{
  "same_prefix_as_mcd1_abs.csv": {
    "concentration_mol_L": 0.01,
    "pathlength_cm": 1
  },
  "same_prefix_as_mcd2_abs.csv": {
    "concentration_mol_L": 0.005,
    "pathlength_cm": 1
  }
}
```

### How to Use

1. **Place your CSV files in a directory**: Ensure your CSV files are named appropriately to include `pos`, `neg`, and `abs` to indicate positive, negative, and absorption data respectively.

2. **Prepare the Configuration Files**:
   - Create `config.json` and `abs_data.json` files in the same directory as the script or specify the correct paths.

3. **Run the Script**:
   - Execute the script by running the following command in your terminal or command prompt:
     ```bash
     python mcd_processing.py
     ```

4. **Select the Files**:
   - A file dialog will appear prompting you to select the CSV files. Select the files you want to process.

5. **Processed Data**:
   - The processed data will be saved in a new directory called `processed_data` within the directory where the original files are located.
   - The script will generate plots for the MCD, Absorption, and MORD data. If configured, it will also plot derivatives and sticks for transition predictions.

### Function Overview

#### `load_json(file_path: str) -> dict`
Loads data from a JSON file.

#### `select_files() -> defaultdict`
Opens a file dialog to select multiple files and tags each as positive, negative, or absorption.

#### `create_output_directory(base_path: str) -> str`
Creates an output directory within the base path.

#### `read_csv_file(filename: str, column_names: list = None) -> pd.DataFrame`
Reads a CSV file into a DataFrame with specified column names if provided.

#### `calculate_differences(positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> tuple`
Calculates differences and standard deviations between positive and negative field data.

#### `interpolate_data(wavelength: pd.Series, R_signed: pd.Series, spline_points: int) -> tuple`
Interpolates data for smoother plotting.

#### `kk_arbspace(omega: np.ndarray, imchi: np.ndarray, alpha: int) -> np.ndarray`
Calculates the real part of susceptibility using Kramers-Kronig relations.

#### `convert_abs_to_extinction(df: pd.DataFrame, filename: str, abs_data: dict, column: str = 'intensity') -> pd.DataFrame`
Converts absorbance to extinction using pathlength and concentration.

#### `scale_sticks(sticks_df: pd.DataFrame, max_absorbance: float, scale_factor: float = 1) -> pd.DataFrame`
Scales the sticks' strengths to be less than the maximum value of the absorbance.

#### `plot_data(mcd_df: pd.DataFrame, abs_df: pd.DataFrame, mord_df: pd.DataFrame, config: dict, sticks_df: pd.DataFrame = None)`
Plots the MCD, Absorption, and MORD data on a single figure with three stacked graphs.

#### `save_data(output_file_path: str, wavelength: pd.Series, R_signed: pd.Series, R_signed_averaged_filled: pd.Series, R_stdev: pd.Series)`
Saves processed data to a CSV file.

#### `process_files(file_dict: defaultdict, config: dict, abs_data: dict)`
Processes each set of files and performs MCD, Absorption, and MORD analysis.

#### `main()`
Main function to load configuration, select files, and process the data.

### Troubleshooting

- **No Files Selected**: Ensure you select the correct CSV files when prompted.
- **File Naming**: Ensure the files are named correctly to include `pos`, `neg`, and `abs`.
- **Configuration Errors**: Check the `config.json` and `abs_data.json` files for correct formatting and valid paths.

### License

Copyright (C) 2024 Samuel Smith

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>

For any further questions or issues, please contact Sam Smith ssmith43@unm.edu


