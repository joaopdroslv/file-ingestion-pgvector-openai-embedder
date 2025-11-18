from code.config.env_variables import VECTOR_DIMENSION
from code.database.connection import get_conn
from typing import Any, Dict, List

from psycopg2.extras import Json


def ensure_database_struct_exists() -> None:
    """Wrapper function to create all tables/database structure."""

    create_embeddings_table()


def create_embeddings_table() -> None:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id          SERIAL PRIMARY KEY,
                doc_id      TEXT UNIQUE,
                content     TEXT,
                metadata    JSONB,
                embedding   vector({VECTOR_DIMENSION})
            );
            """
            )
            conn.commit()


def upsert_document(
    doc_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embeddings (doc_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id)
                DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding;
                """,
                (doc_id, content, Json(metadata), embedding),
            )
            conn.commit()
