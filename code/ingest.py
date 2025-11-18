import argparse
import os
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

    for text in texts:

        chunks = text_splitter.split_text(text)
        embeddings = embedder_model.embed_documents(chunks)
        document_id = str(uuid.uuid4())

        for i, chunk in enumerate(chunks):

            print(f">>> Ingesting chunk | {i + 1}")

            embedded_chunk = embeddings[i]
            chunk_id = str(uuid.uuid4())
            metadata = {"document_id": document_id, "chunk_index": i}

            if len(embedded_chunk) != VECTOR_DIMENSION:
                raise RuntimeError(
                    f"Embedding with dimension greater than {VECTOR_DIMENSION}."
                )

            upsert_document(chunk_id, chunk, metadata, embedded_chunk)

    print(f">>> Ingested {len(texts)} documents (expanded to chunks and stored).")


def main() -> None:

    ensure_database_struct_exists()

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The path to the PDF/DOCX file.")
    parser.add_argument(
        "--all",
        help="The path to a folder containing multiple documents to be ingested.",
    )
    args = parser.parse_args()

    contents = []

    if args.file:
        print(">>> Reading file |", args.file)
        file_content = load_file(args.file)
        contents.append(file_content)

    if args.all:
        folder = args.all

        if not os.path.isdir(folder):
            parser.error(f"--all must be a valid folder, got: {folder}")

        print(">>> Reading all documents in folder |", folder)

        for f in os.listdir(folder):
            full_path = os.path.join(folder, f)

            if not os.path.isfile(full_path):
                continue

            if f.lower().endswith((".pdf", ".docx")):
                print(">>> Reading file |", full_path)

                file_content = load_file(full_path)
                contents.append(file_content)

    if not contents:
        parser.error("Nothing to ingest, ending script.")

    ingest(contents)


if __name__ == "__main__":
    main()
