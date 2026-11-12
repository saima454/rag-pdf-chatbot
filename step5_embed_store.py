"""
Step 7-8: Google embeddings  ->  save in Qdrant.

Usage:
    uv run python step5_embed_store.py
Input : output/chunks.json

Qdrant connection (set in .env):
    - QDRANT_URL empty      -> LOCAL mode: data saved in the folder  qdrant_db/  (no server needed)
    - QDRANT_URL=http://localhost:6333            -> your own Qdrant server (Docker)
    - QDRANT_URL=https://xxxx.cloud.qdrant.io:6333 + QDRANT_API_KEY=...  -> Qdrant Cloud

Re-running is safe: every chunk gets a fixed id (uuid made from its chunk id), so upsert
overwrites instead of duplicating.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

CHUNKS_FILE = Path("output/chunks.json")
LOCAL_DB_PATH = "qdrant_db"
COLLECTION = "pdf_chunks"
BATCH_SIZE = 50            # chunks per embedding request (API limit is 100)
EMBED_DIM = 768            # must match the vector size of the Qdrant collection


# ---------------------------------------------------------------- Google embeddings
def get_embed_client_and_model():
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not found. Put it in the .env file.")
    model = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    return genai.Client(api_key=api_key), model


def embed_texts(client, model: str, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"):
    """Return one embedding (list of floats) per text. Retries on temporary API errors.
    Use task_type="RETRIEVAL_QUERY" later when embedding the user's question."""
    from google.genai import types

    for attempt in range(4):
        try:
            res = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=EMBED_DIM),
            )
            return [e.values for e in res.embeddings]
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  embedding error ({e}); retry in {wait}s")
            time.sleep(wait)
    sys.exit("Embedding failed after several retries.")


# ---------------------------------------------------------------- Qdrant
def get_qdrant_client():
    """Step 1 of storing: create the connection."""
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient

    load_dotenv()
    url = os.getenv("QDRANT_URL", "").strip()
    if url:
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        print(f"Connecting to Qdrant server: {url}")
        return QdrantClient(url=url, api_key=api_key)
    print(f"Using local Qdrant storage in folder: {LOCAL_DB_PATH}/")
    return QdrantClient(path=LOCAL_DB_PATH)


def ensure_collection(qdrant, name: str = COLLECTION, size: int = EMBED_DIM):
    """Step 2: create the collection (like a table) if it does not exist yet."""
    from qdrant_client import models

    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )
        print(f"Created collection '{name}' (size={size}, cosine)")


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be an integer or a UUID, so make a stable UUID from our chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def embed_and_store(chunks: list[dict]) -> int:
    from qdrant_client import models

    embed_client, model = get_embed_client_and_model()
    qdrant = get_qdrant_client()
    ensure_collection(qdrant)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors = embed_texts(embed_client, model, [c["text"] for c in batch])
        # Step 3: upsert points = vector + payload (the text and metadata)
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id(c["id"]),
                    vector=vec,
                    payload={"text": c["text"], "source": c["source"], "page": c["page"],
                             "chunk_index": c["chunk_index"], "chunk_id": c["id"]},
                )
                for c, vec in zip(batch, vectors)
            ],
        )
        print(f"Stored chunks {start + 1}-{start + len(batch)} of {len(chunks)}")
        time.sleep(1)  # gentle on free-tier rate limits
    return qdrant.count(collection_name=COLLECTION, exact=True).count


if __name__ == "__main__":
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    total = embed_and_store(chunks)
    print(f"\nDone. Qdrant collection '{COLLECTION}' now holds {total} points")
