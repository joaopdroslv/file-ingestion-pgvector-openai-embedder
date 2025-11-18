import json
from typing import Any, Dict, List

import psycopg2

from code.models.embedder_model import embedder_model
from code.config.env_variables import DATABASE_URL


def search_pgvector(query: str, limit: int = 5, metadata_filter: Dict = None) -> List[Dict[str, Any]]:

    query_vector = embedder_model.embed_query(query)

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
