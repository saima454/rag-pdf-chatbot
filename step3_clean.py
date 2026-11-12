"""
Step 5: Cleaning  (rule-based first, then Gemini LLM)

Usage:
    python step3_clean.py              # rules + LLM
    python step3_clean.py --no-llm     # rules only (no API key needed)
Input : output/extracted_pages.json
Output: output/cleaned_pages.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

IN_FILE = Path("output/extracted_pages.json")
OUT_FILE = Path("output/cleaned_pages.json")
IMAGE_MARKER = "[Image text]"
REPEAT_RATIO = 0.4   # a line appearing on >=40% of pages = header/footer
MIN_PAGES_FOR_REPEAT = 3


# ---------------------------------------------------------------- helpers
def norm(s: str) -> str:
    """Normalised form used only for comparing text (lowercase, no punctuation/extra spaces)."""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def collapse_spaces(text: str) -> str:
    text = text.replace(" ", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)               # many spaces -> one
    text = re.sub(r" *\n *", "\n", text)               # spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)             # many blank lines -> one
    return text.strip()


def find_repeated_lines(pages: list[dict]) -> set[str]:
    """Headers/footers: lines that repeat across many pages (page numbers ignored)."""
    if len(pages) < MIN_PAGES_FOR_REPEAT:
        return set()
    counter = Counter()
    for p in pages:
        seen = {norm(re.sub(r"\d+", "", l)) for l in p["combined_text"].splitlines() if l.strip()}
        counter.update(x for x in seen if x)
    limit = max(2, int(len(pages) * REPEAT_RATIO))
    return {line for line, c in counter.items() if c >= limit}


# ---------------------------------------------------------------- rule-based cleaning
MIN_WORDS_FOR_DEDUPE = 4  # short lines (headings, "Page 2") are never treated as duplicates


def drop_repeated_lines(text: str, repeated: set[str]) -> str:
    lines = [l for l in text.splitlines()
             if norm(re.sub(r"\d+", "", l)) not in repeated]
    return collapse_spaces("\n".join(lines))


def dedupe_sentences(text: str, seen: set[str]) -> tuple[str, int]:
    """Remove sentences already seen earlier (same page or any earlier page).

    Works on sentences, not lines, because pypdf wraps one sentence over several lines.
    The first occurrence is kept; later ones are cut out of the original text.
    """
    removed = 0
    # A short line without end punctuation is a heading: close it with "." so it does
    # not get glued to the next sentence (which would hide duplicates of that sentence).
    def is_heading(line: str) -> bool:
        return 0 < len(line.split()) <= 10 and not re.search(r"[.!?,;:]$", line)

    flat = " ".join(l + "." if is_heading(l) else l
                    for l in (x.strip() for x in text.splitlines()) if l)
    for sent in re.split(r"(?<=[.!?])\s+", flat):
        key = norm(sent)
        if len(key.split()) < MIN_WORDS_FOR_DEDUPE:
            continue
        if key not in seen:
            seen.add(key)
            continue
        # build a regex that matches this sentence even if it is wrapped over lines
        pattern = r"\s+".join(re.escape(w) for w in sent.split())
        text, n = re.subn(pattern + r"\s*", "", text, count=1)
        removed += n
    return collapse_spaces(text), removed


def rule_clean(pages: list[dict]) -> list[dict]:
    repeated = find_repeated_lines(pages)
    print(f"Repeated header/footer patterns found: {len(repeated)}")
    seen: set[str] = set()
    removed_total = 0

    for p in pages:
        # text layer first, then OCR text of each image (so the text-layer copy is the one kept)
        parts = [p["pdf_text"]] + p["ocr_texts"]
        cleaned_parts = []
        for part in parts:
            part = drop_repeated_lines(part, repeated)
            part, n = dedupe_sentences(part, seen)
            removed_total += n
            if part:
                cleaned_parts.append(part)
        p["rule_cleaned_text"] = "\n\n".join(cleaned_parts)

    print(f"Duplicate sentences removed: {removed_total}")
    return pages


# ---------------------------------------------------------------- LLM cleaning
PROMPT = """You are a text-cleaning tool for a document search system.
Clean the text below. Rules:
1. Fix obvious OCR mistakes (broken words, wrong characters) ONLY when you are sure.
2. Remove duplicate or near-duplicate sentences (keep one).
3. Remove extra spaces and fix broken line breaks inside sentences.
4. Keep headings, lists, numbers, emails and phone numbers exactly as they are.
5. NEVER add new information, NEVER summarise, NEVER translate, NEVER explain.
6. Return ONLY the cleaned text, nothing else.

TEXT:
{text}"""


def llm_clean(pages: list[dict]) -> list[dict]:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not found. Put it in the .env file.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    for p in pages:
        src = p["rule_cleaned_text"]
        if not src:
            p["cleaned_text"] = ""
            continue
        result = None
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=PROMPT.format(text=src),
                    config=types.GenerateContentConfig(temperature=0),
                )
                result = (r.text or "").strip()
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  page {p['page']}: API error ({e}); retry in {wait}s")
                time.sleep(wait)

        # Safety net: if the LLM output looks wrong, keep the rule-cleaned text
        if not result or not (0.5 * len(src) <= len(result) <= 1.2 * len(src)):
            print(f"  page {p['page']}: LLM output rejected, using rule-cleaned text")
            result = src
        p["cleaned_text"] = collapse_spaces(result)
        print(f"Page {p['page']:>2}: {len(src)} -> {len(p['cleaned_text'])} chars (LLM)")
        time.sleep(1)  # be gentle with free-tier rate limits
    return pages


if __name__ == "__main__":
    pages = json.loads(IN_FILE.read_text(encoding="utf-8"))
    pages = rule_clean(pages)
    if "--no-llm" in sys.argv:
        for p in pages:
            p["cleaned_text"] = p["rule_cleaned_text"]
    else:
        pages = llm_clean(pages)
    OUT_FILE.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    before = sum(len(p["combined_text"]) for p in pages)
    after = sum(len(p["cleaned_text"]) for p in pages)
    print(f"\nTotal chars: {before} -> {after}. Saved to {OUT_FILE}")
