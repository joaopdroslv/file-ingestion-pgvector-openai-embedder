import argparse
import json
import os
import uuid
from typing import Any, Dict, List

import docx
import pdfplumber
import psycopg2
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg2.extras import Json

load_dotenv()

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION"))
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAPI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not found.")


def search_pgvector(query: str, limit: int = 5, metadata_filter: Dict = None):
    embedder = OpenAIEmbeddings(api_key=OPENAPI_API_KEY)
    query_vector = embedder.embed_query(query)

    def to_pgvector(vec):
        return "[" + ",".join(str(v) for v in vec) + "]"

    query_vec_str = to_pgvector(query_vector)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    where_clauses = []
    params = [query_vec_str]

    if metadata_filter:
        for idx, (key, value) in enumerate(metadata_filter.items(), start=2):
            where_clauses.append(f"metadata->>'{key}' = %s")
            params.append(value)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
        SELECT
            id,
            content,
            metadata,
            embedding <=> %s::vector AS distance
        FROM documents
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT {limit};
    """

    # Add second embedding for ORDER BY
    params.append(query_vec_str)

    cur.execute(sql, params)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {"id": row[0], "content": row[1], "metadata": row[2], "distance": float(row[3])}
        for row in rows
    ]


if __name__ == "__main__":
    query = input("Type something to search: ")

    results = search_pgvector(
        query=query,
        limit=5,
        metadata_filter=None,  # Example: {"category": "some category"}
    )

    print("\n🔍 Results:")
    for r in results:
        print("---------------")
        print(f"ID: {r['id']}")
        print(f"Distance: {r['distance']:.4f}")
        print(f"Metadata: {json.dumps(r['metadata'], indent=2)}")
        print(f"Content:\n{r['content'][:500]}")
        print()
