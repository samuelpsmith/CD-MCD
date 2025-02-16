import os

from tkinter import messagebox

from MCD_process.base.data_processing import process_files
from base.utils import file_handler as fh
from base.utils import logger

#init root logger
logger.init_root_logger("mcd_process_log.txt")
#get logger with module name
logging = logger.get_logger(__name__)


def main():
    try:
        # Construct the absolute path to 'abs_data.json' based on the script's location
        # basically i iassume that abs_data.json, config.json and this script are all in the same directory.
        #i think I also had some issue where if you were running this on mac vs windows it didnt build the directory nmame right. (Case sensitvie?)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_data_path = os.path.join(script_dir, "abs_data.json")
        config_path = os.path.join(script_dir, "config.json")
        abs_data = fh.load_json(abs_data_path)
        config = fh.load_json(config_path)
        file_dict = fh.select_files_processing()
        if file_dict:
            process_files(file_dict, config, abs_data)
        else:
            logging.error("No files were selected or tagged properly.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    main()
