import os
from tkinter import messagebox
from base.data_processing import process_files
from base.utils import file_handler as fh
from base.utils import logger
import rw_db as db

#init root logger
logger.init_root_logger("mcd_process_log.txt")
#get logger with module name
logging = logger.get_logger(__name__)

#Params: String id - LIMS id for sample
#        String name - name of the sample
#        float conc - concentration of the sample
#        float path_length - path_length of the sample
#        float field_strength - field_strength of the sample
#Returns: dict - returns dictionary containing data
#Does: Creates a dictionary at the end of user input
def create_dict(id, name, conc, path_length, field_strength):
    dic = {
        "id" : id,
        "name" : name,
        "concentration_mol_L" : conc,
        "pathlength_cm" : path_length,
        "field_B" : field_strength
    }
    return dic

def main():
    try:
        # Construct the absolute path to 'abs_data.json' based on the script's location
        # basically i iassume that abs_data.json, config.json and this script are all in the same directory.
        #i think I also had some issue where if you were running this on mac vs windows it didnt build the directory nmame right. (Case sensitvie?)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        config = fh.load_json(config_path)
        abs_data = None
        LOAD_FROM_DB = config.get("load_from_db")

        if not LOAD_FROM_DB:
            id = input("Enter ID: ")
            name = input("Enter name: ")
            conc = input("Enter concentration MOL/L: ")
            path_length = input("Enter pathlength mm: ")
            field_B = input("Enter field strength B: ")
            abs_data = create_dict(id,name,float(conc),float(path_length),float(field_B))
        else:
            #Load from db
            abs_data_name = input("Enter LIMS ID: ")
            abs_data = db.get_fields(abs_data_name)
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
