"""
Step 6: Chunking (page-wise, so every chunk keeps its page number).

Usage:
    uv run python step4_chunk.py
Input : output/cleaned_pages.json
Output: output/chunks.json
"""
import json
import re
from pathlib import Path

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 100   # characters repeated between neighbouring chunks

IN_FILE = Path("output/cleaned_pages.json")
OUT_FILE = Path("output/chunks.json")


def _units(text: str, size: int) -> list[str]:
    """Break text into small pieces (paragraph -> sentence -> words), each <= size."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(sent) <= size:
                units.append(sent)
            else:  # very long sentence: cut by words
                cur = ""
                for w in sent.split():
                    if len(cur) + len(w) + 1 > size and cur:
                        units.append(cur)
                        cur = w
                    else:
                        cur = f"{cur} {w}".strip()
                if cur:
                    units.append(cur)
    return units


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, current, cur_len = [], [], 0
    for unit in _units(text, size):
        if current and cur_len + len(unit) + 1 > size:
            chunks.append("\n".join(current))
            # start next chunk with the tail of the previous one (overlap)
            tail, tail_len = [], 0
            for u in reversed(current):
                if tail_len + len(u) + 1 > overlap:
                    break
                tail.insert(0, u)
                tail_len += len(u) + 1
            if tail_len + len(unit) + 1 > size:  # overlap would make chunk too big
                tail, tail_len = [], 0
            current, cur_len = tail, tail_len
        current.append(unit)
        cur_len += len(unit) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def make_chunks(pages: list[dict]) -> list[dict]:
    all_chunks = []
    for p in pages:
        text = p.get("cleaned_text", "")
        for idx, chunk in enumerate(split_text(text)):
            all_chunks.append({
                "id": f"{p['source']}_p{p['page']}_c{idx}",
                "text": chunk,
                "source": p["source"],
                "page": p["page"],
                "chunk_index": idx,
            })
    return all_chunks


if __name__ == "__main__":
    pages = json.loads(IN_FILE.read_text(encoding="utf-8"))
    chunks = make_chunks(pages)
    OUT_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks from {len(pages)} pages "
          f"(min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)} chars)")
    print(f"Saved to {OUT_FILE}")
