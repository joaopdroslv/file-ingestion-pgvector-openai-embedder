import os

from dotenv import load_dotenv

load_dotenv()

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION"))
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY", None)

if not OPENAPI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not found.")
