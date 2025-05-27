from psycopg2 import OperationalError, InterfaceError
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
from singletons.logger import get_logger
import time
import os

load_dotenv(override=True)

_db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=15,
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    user=os.getenv("POSTGRES_USERNAME"),
    password=os.getenv("POSTGRES_PASSWORD"),
    dbname=os.getenv("POSTGRES_DATABASE"),
)


def get_db_connection(retries=2):
    global _db_pool
    last_exception = None
    logger = get_logger()

    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _db_pool.getconn()
            # Perform a quick check to see if the connection is alive
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            # If the check passes, return the connection
            # Rollback any implicit transaction started by SELECT 1 if not in autocommit
            if not conn.autocommit:
                conn.rollback()
            logger.info(
                f"Successfully obtained and validated DB connection (attempt {attempt + 1}).")
            return conn
        except (OperationalError, InterfaceError) as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt + 1} to get/validate DB connection failed: {e}")
            if conn:
                # If we got a connection but it's bad, put it back to be closed by the pool
                _db_pool.putconn(conn, close=True)
            if attempt < retries:
                logger.info("Retrying to get a new DB connection...")
                time.sleep(0.5)  # Small delay before retrying
            else:
                logger.error("Max retries reached for getting DB connection.")
                # Fall through to raise the last_exception outside the loop
        except Exception as e:  # Catch other potential errors from getconn() or the check
            last_exception = e
            logger.error(
                f"Attempt {attempt + 1}: Unexpected error during DB connection retrieval/validation: {e}", exc_info=True)
            if conn:
                # Ensure problematic connection is closed
                _db_pool.putconn(conn, close=True)
            # Fall through to raise the last_exception if this is the last attempt

    # If all retries failed
    if last_exception:
        raise last_exception
    else:
        # This case should ideally not be reached if retries > 0 and an error always occurs
        raise OperationalError(
            "Failed to obtain a valid database connection after multiple attempts.")


def release_db_connection(conn):
    global _db_pool
    _db_pool.putconn(conn)
