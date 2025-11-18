from code.database.connection import get_conn
from code.modules.embeddings import get_all_embeddings_from_a_document

if __name__ == "__main__":

    with get_conn() as conn:
        with conn.cursor() as cursor:

            sql = """
                SELECT metadata->>'document_id'
                FROM embeddings
                LIMIT 1;
            """

            cursor.execute(sql)
            row = cursor.fetchone()

            if not row:
                raise RuntimeError("Not embedding available.")

            document_id = row[0] if row else None

    result = get_all_embeddings_from_a_document(document_id=document_id)

    print(f'>>> Displaying all available embeddings for document_id="{document_id}"')

    for embedding in result.embeddings:
        print(f">>> Embedding with ID [ {embedding.id} ]")
        print(f"\tContent Length [ {len(embedding.content)} ]")
        print(">>> Embedding Metadata")
        print(f"\tChunk Index [ {embedding.metadata.chunk_index} ]")
