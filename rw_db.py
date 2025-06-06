import os.path
import sqlite3 as db
import base.utils.logger as logger

#maybe will log database connections
logging = logger.get_logger(__name__)

#Parmas: string lims_id - LIMS id of the sample to get the fields for
#Return: dict (id, name, concentration_mol_L, pathlength_cm, field_B)- dictionary of the field
#Does: Gets the fields of a sample in the database
def get_fields(lims_id):
    connection = None
    fields = None
    try:
        connection = db.connect("processing_db.db")
        print("connected to db")
        cur = connection.cursor()
        res = cur.execute("SELECT * FROM SAMPLES WHERE id = ?", (lims_id,))
        print("collected fields")
        fields = res.fetchone()
    except Exception as e:
        print("Failed to find lims ID or connect to database: exception -> "+str(e))
    finally:
        connection.close()
        print("connection closed")
        return create_dict(fields)

#Params: None
#Returns: Void
#Does: Creates a local database to store LIMS sample info
def create_db():
    if os.path.exists("processing_db.db"):
        print("Database already created")
        return
    connection = None
    try:
        connection = db.connect("processing_db.db")
        print("created and connected to db")
        cur = connection.cursor()
        cur.execute("CREATE TABLE SAMPLES(id, name, concentration, pathlength, field)")
    except:
        print("unexpected error")
    finally:
        connection.close()
        print("connection closed")
#Params: string lims_ID - LIMS id
#        string name - name of sample
#        float conc - concentration of sample
#        float pl - path length of sample
#        float field - field strength applied to sample
#Returns: Void
#Does: Adds a sample to the local database
def populate_table(lims_ID, name, conc, pl, field):
    connection = None
    try:
        connection = db.connect("processing_db.db")
        print("connected to db to populate table")
        cur = connection.cursor()
        query = cur.execute("SELECT * FROM SAMPLES WHERE id = ?", (lims_ID,))
        res = query.fetchone()
        if res is not None:
            print("sample already exists in database")
        else:
            cur.execute("INSERT INTO SAMPLES VALUES(?,?,?,?,?)",(lims_ID, name, conc,pl,field))
            connection.commit()
    finally:
        connection.close()
        print("connection closed")
#Params: tuple query - tuple containing sample fields
#Retrun: dict - Dictionary of the sample fields
#Does: Creates a dictionary of the sample fields out of a sql lite query of the local sample database
def create_dict(query):
    dic = {
        "id" : query[0],
        "name" : query[1],
        "concentration_mol_L" : query[2],
        "pathlength_cm" : query[3],
        "field_B" : query[4]
    }
    return dic
#Params: None
#Returns: Void
#Does: Calls for user input to enter a sample
def enter_samples():
    while True:
        id = input("Enter LIMS ID: ")
        name = input("Enter name: ")
        conc = input("Enter concentration: ")
        pl = input("Enter pathlength: ")
        field = input("Enter field strength: ")
        populate_table(id,name,float(conc),float(pl),float(field))
        cont = input("Would you like to enter more (y/n)? ")
        if cont == "n":
            break

#Params: string lims_ID - LIMS id od the sample to delete
#Returns: Void
#Does: Deletes a sample from the local database
def delete_field(lims_ID):
    connection = None
    try:
        connection = db.connect("processing_db.db")
        print("Connected to db for delete")
        cur = connection.cursor()
        cur.execute("DELETE FROM SAMPLES WHERE id = ?", (lims_ID,))
    finally:
        print("closed connection to db for delete")
        connection.close()
if __name__ == "__main__":
    enter_samples()