import logging
import psycopg2
from dotenv import load_dotenv
import os


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def connect_to_db():
    load_dotenv()
    conn = psycopg2.connect(
        host="localhost",
        database=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        port=5432,
    )
    return conn


def get_cursor(conn):
    return conn.cursor()
