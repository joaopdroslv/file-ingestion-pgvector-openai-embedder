from code.config.env_variables import DATABASE_URL
from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector

# @contextmanager
# def get_conn():
#     conn = psycopg2.connect(DATABASE_URL)
#     cursor = conn.cursor()
#     try:
#         yield cursor
#         conn.commit()
#     except Exception:
#         conn.rollback()
#         raise
#     finally:
#         cursor.close()
#         conn.close()


def get_conn():
    connection =  psycopg2.connect(DATABASE_URL)
    register_vector(connection)
    return connection
