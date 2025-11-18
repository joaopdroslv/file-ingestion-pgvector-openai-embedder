import os

from langchain_openai import OpenAIEmbeddings

OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAPI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not found.")

embedder_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAPI_API_KEY,
)
