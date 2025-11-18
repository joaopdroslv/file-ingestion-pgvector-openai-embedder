from code.config.env_variables import VECTOR_DIMENSION
from code.database.connection import get_conn
from code.models.embedder_model import embedder_model
from code.schemas.embeddings import (
    EmbeddingMetadata,
    SimplifiedEmbedding,
    SimplifiedEmbeddings,
)
from typing import Any, Dict, List

from pgvector import Vector
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
                chunk_id    TEXT UNIQUE,
                content     TEXT,
                metadata    JSONB,
                embedding   vector({VECTOR_DIMENSION})
            );
            """
            )
            conn.commit()


def upsert_document(
    chunk_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embeddings (chunk_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id)
                DO UPDATE
                    SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                """,
                (chunk_id, content, Json(metadata), Vector(embedding)),
            )
            conn.commit()


def get_all_embeddings_from_a_document(document_id: str) -> SimplifiedEmbeddings:

    sql = """
        SELECT
            id,
            content,
            metadata
        FROM embeddings
        WHERE metadata->>'document_id' = %s
        ORDER BY (metadata->>'chunk_index')::INT ASC
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, [document_id])
            embeddings = cursor.fetchall()

    return [
        SimplifiedEmbedding(
            id=row[0],
            content=row[1],
            metadata=EmbeddingMetadata.from_dict(row[2]),
        )
        for row in embeddings
    ]


def get_embedding_cousine_distance(
    query: str, limit: int = 5, metadata: Dict = None
) -> List[Dict[str, Any]]:

    query_vector = Vector(embedder_model.embed_query(query))

    with get_conn() as conn:
        with conn.cursor() as cursor:

            where_clauses = []
            params = [query_vector]

            if metadata:
                for key, value in metadata.items():
                    where_clauses.append("metadata->>%s = %s")
                    params.extend([key, value])

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            sql = f"""
                SELECT
                    id,
                    content,
                    metadata,
                    embedding <=> %s AS distance
                FROM embeddings
                {where_sql}
                ORDER BY distance ASC
                LIMIT %s;
            """

            params.append(limit)

            cursor.execute(sql, params)
            embeddings = cursor.fetchall()

    return [
        {"id": emb[0], "content": emb[1], "metadata": emb[2], "distance": float(emb[3])}
        for emb in embeddings
    ]
