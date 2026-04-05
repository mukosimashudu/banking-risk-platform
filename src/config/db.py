from __future__ import annotations

import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")


def build_engine() -> Engine | None:
    """
    Build and return a SQLAlchemy engine for Azure SQL Server.

    Returns None if required environment variables are missing so that the app
    can still start and expose useful health/debug messages instead of crashing
    at import time.
    """
    required = {
        "DB_SERVER": DB_SERVER,
        "DB_DATABASE": DB_DATABASE,
        "DB_USERNAME": DB_USERNAME,
        "DB_PASSWORD": DB_PASSWORD,
    }

    missing = [key for key, value in required.items() if not value]
    if missing:
        print(
            "⚠️ Database engine not created. Missing environment variables: "
            + ", ".join(missing)
        )
        return None

    connection_string = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_USERNAME};"
        f"PWD={DB_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

    params = quote_plus(connection_string)

    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        fast_executemany=True,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )


engine = build_engine()