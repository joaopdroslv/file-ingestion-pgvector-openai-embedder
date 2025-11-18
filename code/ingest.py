import argparse
import uuid
from typing import Any, Dict, List

import docx
import pdfplumber
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg2.extras import Json

from code.config.env_variables import VECTOR_DIMENSION, DATABASE_URL
from code.models.embedder_model import embedder_model


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
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
):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (doc_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id)
                DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding;
                """,
                (doc_id, content, Json(metadata), embedding),
            )
            conn.commit()


def read_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    return "\n".join(texts)


def read_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def load_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    elif path.lower().endswith(".docx") or path.lower().endswith(".doc"):
        return read_docx(path)
    else:
        raise RuntimeError("File format not supported.")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
)


def ingest(texts: List[str]):

    ensure_table()

    for original_index, text in enumerate(texts):
        chunks = text_splitter.split_text(text)

        embeddings = embedder_model.embed_documents(chunks)

        for i, chunk in enumerate(chunks):
            print(f">>> Ingesting chunk | {i + 1}")
            embedded_chunk = embeddings[i]

            if len(embedded_chunk) != VECTOR_DIMENSION:
                raise RuntimeError(
                    f"Embedding with dimension greater than {VECTOR_DIMENSION}."
                )

            doc_id = f"{uuid.uuid4()}"
            metadata = {"source_index": original_index, "chunk_index": i}

            upsert_document(doc_id, chunk, metadata, embedded_chunk)

    print(f">>> Ingested {len(texts)} documents (expanded to chunks and stored).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The path to the PDF/DOCX file.")
    args = parser.parse_args()

    contents = []
    if args.file:
        print(">>> Reading file |", args.file)
        file_content = load_file(args.file)
        contents.append(file_content)

    if not contents:
        parser.error("A --file argument with a valid file path is required.")

    ingest(contents)


if __name__ == "__main__":
    main()
