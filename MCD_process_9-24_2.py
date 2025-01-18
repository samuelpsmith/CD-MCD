import os
import pandas as pd
import numpy as np
from tkinter import messagebox
from collections import defaultdict
from base.tools import file_reader as fr, logger
from base import data_processing as dpro, data_plotting as dplt

#get logger with module name
logging = logger.get_logger(__name__)

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)
print(f"Current working directory: {os.getcwd()}")


def save_data(output_file_path, mcd_df, abs_df_copy, mord_df, sticks_df=None):
    # Merge MCD and Absorption data
    # considering renaming to MCD_process, abs, mord. But doing so would break some of the modularity of the code.

    print("sticks_df in savedata:", sticks_df)
    merged_df = pd.merge(
        mcd_df.rename(columns={"R_signed_extinction": "MCD_process"}),
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

            positive_df = fr.read_csv_file(pos_file, column_names_pos)
            negative_df = fr.read_csv_file(neg_file, column_names_neg)
            abs_df = fr.read_csv_file(abs_file, column_names_abs)
            sticks_df = (
                fr.read_csv_file(sticks_file, column_names_sticks) if sticks_file else None
            )

            if (
                    positive_df is not None
                    and negative_df is not None
                    and abs_df is not None
            ):
                try:
                    abs_df_copy = abs_df.copy()
                    # pass by ref vs pass by value.

                    # I dont know that we ever really want to look at this in units of abs? consider changing?

                    if config["convert_to_extinction"]:
                        print("convert to extinction is true (abs)")
                        abs_df_copy = dpro.convert_abs_to_extinction(
                            abs_df_copy,
                            os.path.basename(abs_file),
                            abs_data,
                            ["intensity"],
                        )

                    x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev = (
                        dpro.calculate_differences(positive_df, negative_df)
                    )

                    mcd_df = pd.DataFrame(
                        {
                            "wavelength": positive_df["wavelength"],
                            "R_signed": R_signed,
                            "std_dev": R_stdev,
                        }
                    )

                    if config["convert_to_extinction"]:
                        print("convert to extinction is true (MCD_process)")
                        mcd_df = dpro.convert_abs_to_extinction(
                            mcd_df,
                            os.path.basename(abs_file),  # whats goin on here?
                            abs_data,
                            ["R_signed", "std_dev"],
                        )

                    R_signed_averaged_filled = (
                        mcd_df[
                            "R_signed_extinction"]  # this is bad. This is a bad idea and we gotta not do this anymore.
                        .rolling(window=3, center=True)
                        .mean()
                        .fillna(mcd_df["R_signed_extinction"])
                    )  # we DO need to fill in NaNs, but we ought not to rolling average data anymore.

                    wavenumber_cm1 = 1e7 / mcd_df[
                        "wavelength"].values  # gotta convert to wavenumber to do kramers kronig.
                    mord_spectrum = dpro.kk_arbspace(
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
                    output_dir = fr.create_output_directory(base_path)
                    output_file_path = os.path.join(
                        output_dir, base_name + "processed.csv"
                    )
                    print("outputting to:", output_file_path)

                    if sticks_df is not None:
                        max_absorbance = abs_df_copy["intensity_extinction"].max()
                        sticks_df = dpro.scale_sticks(sticks_df, max_absorbance)
                        sticks_column = sticks_df["scaled_strength"]
                    else:
                        sticks_column = pd.Series(
                            [None] * len(mcd_df["wavelength"]), index=mcd_df.index
                        )

                    # fitting goes here. fitted_params = fit_peaks_seperately_old(abs_df_copy, mcd_df, column='intensity_extinction', height_percent=1, lorentz_frac=0.5)

                    # change to new plot data function
                    dplt.plot_data_old(
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
        abs_data = fr.load_json(abs_data_path)
        config = fr.load_json(config_path)
        file_dict = fr.select_files()
        if file_dict:
            process_files(file_dict, config, abs_data)
        else:
            logging.error("No files were selected or tagged properly.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    main()
