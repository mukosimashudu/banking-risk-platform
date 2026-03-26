import os
from dotenv import load_dotenv

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_DRIVER = os.getenv("DB_DRIVER")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")