from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingMetadata:
    chunk_index: int
    document_id: str

    @staticmethod
    def from_dict(data: dict) -> "EmbeddingMetadata":
        return EmbeddingMetadata(
            chunk_index=data["chunk_index"],
            document_id=data["document_id"],
        )


@dataclass
class SimplifiedEmbedding:
    """A simplified embedding version with ID, content and metadata only."""

    id: int
    content: str
    metadata: EmbeddingMetadata


@dataclass
class SimplifiedEmbeddings:
    embeddings: List[SimplifiedEmbedding]


@dataclass
class ComparedEmbedding(SimplifiedEmbedding):
    """Embedding with cousine distance after comparsion with a query."""

    distance: float
