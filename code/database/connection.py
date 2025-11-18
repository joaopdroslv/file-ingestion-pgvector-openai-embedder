import psycopg2

from code.config.env_variables import DATABASE_URL


def get_conn():
    return psycopg2.connect(DATABASE_URL)
