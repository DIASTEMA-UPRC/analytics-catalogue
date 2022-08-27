import os

from pymongo import MongoClient

MONGO_HOST = os.getenv("MONGO_HOST", "0.0.0.0")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))

def get_db_object(host: int=MONGO_HOST, port: int=MONGO_PORT):
    mongo = MongoClient(host, port)
    obj = mongo["Diastema"]["Analytics"]
    return obj
