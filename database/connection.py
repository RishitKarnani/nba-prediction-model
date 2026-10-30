"""Database connection module for NBA Prediction using SQLAlchemy and pyodbc."""

import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Load environment variables
load_dotenv()

DB_SERVER = os.getenv("DB_SERVER", "(localdb)\\MSSQLLocalDB")
DB_NAME = os.getenv("DB_NAME", "NBA_Prediction")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
DB_TRUSTED = os.getenv("DB_TRUSTED_CONNECTION", "yes")
DB_TRUST_CERT = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "yes")


def get_connection_url() -> str:
    """Build the SQLAlchemy connection URL with ODBC parameters."""
    odbc_str = (
        f"Driver={{{DB_DRIVER}}};"
        f"Server={DB_SERVER};"
        f"Database={DB_NAME};"
        f"Trusted_Connection={DB_TRUSTED};"
        f"TrustServerCertificate={DB_TRUST_CERT};"
    )
    params = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={params}"


def get_engine() -> Engine:
    """Create and return a SQLAlchemy Engine."""
    url = get_connection_url()
    return create_engine(url, fast_executemany=True)


def test_connection() -> bool:
    """Test SQL Server connectivity and return True if successful."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS is_alive")).fetchone()
            return result[0] == 1
    except Exception as exc:
        print(f"Database connection error: {exc}")
        return False


if __name__ == "__main__":
    print("Testing SQL Server connection...")
    if test_connection():
        print(f"Successfully connected to database '{DB_NAME}' on '{DB_SERVER}'!")
    else:
        print("Failed to connect.")

