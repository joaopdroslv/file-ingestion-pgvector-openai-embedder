import argparse
import uuid
from code.config.env_variables import VECTOR_DIMENSION
from code.models.embedder_model import embedder_model
from code.modules.embeddings import ensure_database_struct_exists, upsert_document
from code.modules.files import load_file
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
)


def ingest(texts: List[str]) -> None:

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


def main() -> None:

    ensure_database_struct_exists()

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
