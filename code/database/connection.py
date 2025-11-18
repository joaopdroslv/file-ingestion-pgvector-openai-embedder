from code.config.env_variables import DATABASE_URL

import psycopg2


def get_conn():
    return psycopg2.connect(DATABASE_URL)
