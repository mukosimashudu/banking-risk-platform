import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine


DB_SERVER = os.getenv("DB_SERVER", "").strip()
DB_DATABASE = os.getenv("DB_DATABASE", "").strip()
DB_USERNAME = os.getenv("DB_USERNAME", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server").strip()

missing = [
    name for name, value in {
        "DB_SERVER": DB_SERVER,
        "DB_DATABASE": DB_DATABASE,
        "DB_USERNAME": DB_USERNAME,
        "DB_PASSWORD": DB_PASSWORD,
    }.items()
    if not value
]

if missing:
    raise ValueError(f"Missing required database environment variables: {', '.join(missing)}")

connection_string = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER},1433;"
    f"DATABASE={DB_DATABASE};"
    f"UID={DB_USERNAME};"
    f"PWD={DB_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

params = quote_plus(connection_string)

engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    pool_pre_ping=True,
    pool_recycle=1800,
    fast_executemany=True,
)