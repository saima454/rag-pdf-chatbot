# PDF RAG Chatbot - Ingestion Phase

Pipeline: **PDF -> pypdf text + OCR image text -> combine -> clean (rules + Gemini) -> chunk -> Google embeddings -> Qdrant**

Retrieval + chatbot comes in the next phase.

## 1. One-time setup (Windows + uv)

1. **Install Tesseract OCR** (UB Mannheim Windows installer). Default path:
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
2. Open `step2_extract.py`, find this line near the top and **remove the `#`**:
   ```python
   # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
   (Fix the path if you installed somewhere else.)
3. In the VS Code terminal, inside this folder:
   ```
   uv sync
   ```
4. Copy `.env.example` to `.env` and paste your Gemini API key after `GEMINI_API_KEY=`.
   Never share the key or commit `.env`.
5. **Qdrant** - choose one in `.env`:
   - Local (easiest, no Docker): leave `QDRANT_URL=` empty. Data is saved in the `qdrant_db/` folder.
   - Your own server: `QDRANT_URL=http://localhost:6333` (e.g. `docker run -p 6333:6333 qdrant/qdrant`).
   - Qdrant Cloud: `QDRANT_URL=https://xxxx.cloud.qdrant.io:6333` and `QDRANT_API_KEY=...`.

## 2. Test step by step (recommended the first time)

```
uv run python step2_extract.py data/rag_test_10pages.pdf   # pypdf + OCR  -> output/extracted_pages.json
uv run python step3_clean.py --no-llm                      # rules only   -> output/cleaned_pages.json
uv run python step3_clean.py                               # rules + Gemini
uv run python step4_chunk.py                               # -> output/chunks.json
uv run python step5_embed_store.py                         # embeddings -> qdrant_db/
uv run python check_db.py                                  # look inside the database
```

## 3. Or everything in one command

```
uv run python run_ingestion.py data/rag_test_10pages.pdf
uv run python run_ingestion.py data/rag_test_10pages.pdf --no-llm
```

## What to check

| Step | Good result |
|---|---|
| extract | 10 pages; pages 2, 4, 5, 7, 9 show `images_with_text=1` |
| clean (rules) | "Duplicate sentences removed: 3" |
| clean (LLM) | `Page N: X -> Y chars (LLM)` for each page; Y slightly smaller than X. "LLM output rejected" means the safety net kept the rule-cleaned text |
| chunk | about 10-13 chunks, max <= 800 chars |
| embed | "Created collection 'pdf_chunks'", "Stored chunks ...", then "holds N points" |
| check_db | 12 points; vectors of 768 numbers; payload with text, source, page |

## Files

- `step2_extract.py` - pypdf text + OCR of images, page-wise
- `step3_clean.py` - header/footer + duplicate removal, space cleanup, Gemini cleaning
- `step4_chunk.py` - 800-char chunks, 100 overlap, page number kept
- `step5_embed_store.py` - Google embeddings (768 dims) -> Qdrant (safe to re-run)
- `run_ingestion.py` - all steps in one go
- `check_db.py` - inspect the database

## Common problems

- `TesseractNotFoundError` -> step 1/2 above (install + path line).
- `API key not valid` -> check `.env` (no quotes, no spaces).
- `model not found` -> change `GEMINI_MODEL` / `EMBED_MODEL` in `.env` to a current model from Google AI Studio.
- `429` / quota errors -> free-tier limit; wait a minute and re-run (embedding step is safe to repeat).
- If you change the embedding model or dimension later, delete the `qdrant_db/` folder (or the Qdrant collection) and ingest again; a collection cannot mix vector sizes.
- Local Qdrant allows only one program at a time: close any other script using `qdrant_db/` before running.
