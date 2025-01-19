from collections import defaultdict
import logger
import json
import os
import tkinter as tk
import re
import pandas as pd
from tkinter import filedialog, messagebox


#get logger
logging = logger.get_logger(__name__)

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
