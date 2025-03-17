import os.path
import sqlite3 as db
import base.utils.logger as logger

#maybe will log database connections
logging = logger.get_logger(__name__)

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
    finally:
        connection.close()
        print("connection closed")
        return create_dict(fields)
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
def create_dict(query):
    dic = {
        "id" : query[0],
        "name" : query[1],
        "concentration_mol_L" : query[2],
        "pathlength_cm" : query[3],
        "field_B" : query[4]
    }
    return dic
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
if __name__ == "__main__":
    enter_samples()