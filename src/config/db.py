from sqlalchemy import create_engine
from urllib.parse import quote_plus
from src.config.settings import DB_SERVER, DB_DATABASE, DB_DRIVER

# Build ODBC connection string
connection_string = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_DATABASE};"
    "Trusted_Connection=yes;"
)

# Encode connection string properly
params = quote_plus(connection_string)

# Create SQLAlchemy engine
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    fast_executemany=True
)