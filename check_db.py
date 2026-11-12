"""
Look inside Qdrant to confirm ingestion worked.

    uv run python check_db.py
"""
from step5_embed_store import COLLECTION, get_qdrant_client

qdrant = get_qdrant_client()
if not qdrant.collection_exists(COLLECTION):
    raise SystemExit(f"Collection '{COLLECTION}' does not exist yet. Run the ingestion first.")

print(f"Total points stored: {qdrant.count(collection_name=COLLECTION, exact=True).count}\n")

points, _ = qdrant.scroll(collection_name=COLLECTION, limit=3,
                          with_payload=True, with_vectors=True)
for p in points:
    pl = p.payload
    print(f"id       : {p.id}")
    print(f"chunk_id : {pl['chunk_id']}  (page {pl['page']}, source {pl['source']})")
    print(f"vector   : {len(p.vector)} numbers, first 4 = {p.vector[:4]}")
    print(f"text     : {pl['text'][:150]!r}...\n")
