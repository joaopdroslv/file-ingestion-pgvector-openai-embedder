import json

from code.modules.embeddings import get_embedding_cousine_distance


if __name__ == "__main__":
    query = input("Type something to search: ")

    results = get_embedding_cousine_distance(
        query=query,
        limit=5,
        metadata=None,  # Example: {"category": "some category"}
    )

    print("\n🔍 Results:")
    for r in results:
        print("---------------")
        print(f"ID: {r['id']}")
        print(f"Distance: {r['distance']:.4f}")
        print(f"Metadata: {json.dumps(r['metadata'], indent=2)}")
        print(f"Content:\n{r['content'][:500]}")
        print()
