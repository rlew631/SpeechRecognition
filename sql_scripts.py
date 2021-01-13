"""
These functions are used to work with storing/loading the data into sql
"""
import sqlite3
from sqlite3 import Error

import numpy as np

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def encode_array_for_sql(array):
    """ takes in a numpy array and outputs a string to store in sql """
    return array.tobytes().decode('ISO-8859-1')

def decode_array_from_sql(string,x,y):
    """ takes in the string to turn into an array and its dimensions """
    return np.frombuffer(string.encode('ISO-8859-1')).reshape(y,x)