"""
Run the complete ingestion pipeline in one command.

    uv run python run_ingestion.py data/rag_test_10pages.pdf
    uv run python run_ingestion.py data/rag_test_10pages.pdf --no-llm     # skip Gemini cleaning
"""
import json
import sys
from pathlib import Path

from step2_extract import extract_pdf
from step3_clean import rule_clean, llm_clean
from step4_chunk import make_chunks
from step5_embed_store import embed_and_store


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("Usage: uv run python run_ingestion.py <file.pdf> [--no-llm]")
    pdf_path = args[0]
    if not Path(pdf_path).exists():
        sys.exit(f"File not found: {pdf_path}")

    print("== 1/4 Extract (pypdf + OCR) ==")
    pages = extract_pdf(pdf_path)
    Path("output").mkdir(exist_ok=True)
    Path("output/extracted_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== 2/4 Clean ==")
    pages = rule_clean(pages)
    if "--no-llm" in sys.argv:
        for p in pages:
            p["cleaned_text"] = p["rule_cleaned_text"]
    else:
        pages = llm_clean(pages)

    Path("output/cleaned_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== 3/4 Chunk ==")
    chunks = make_chunks(pages)
    Path("output/chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(chunks)} chunks created")

    print("\n== 4/4 Embed + store in Qdrant ==")
    total = embed_and_store(chunks)
    print(f"\nIngestion finished. Database holds {total} chunks.")


if __name__ == "__main__":
    main()
