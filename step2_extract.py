"""
Step 2-4: Page-wise extraction  ->  pypdf text + OCR text (images)  ->  combine.

Usage:
    python step2_extract.py path/to/file.pdf
Output:
    output/extracted_pages.json   (page-wise data, input for the cleaning step)
"""
import io
import json
import sys
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
from pypdf import PdfReader

# Windows users: uncomment and set the path where Tesseract is installed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_IMG_SIDE = 50  # ignore tiny images (icons, lines, logos)


def extract_pdf_text(page) -> str:
    """Text layer of the page (pypdf)."""
    return (page.extract_text() or "").strip()


def ocr_image(data: bytes) -> str:
    """Run OCR on one image. Grayscale + upscale small images for better accuracy."""
    img = Image.open(io.BytesIO(data))
    if min(img.size) < MIN_IMG_SIDE:
        return ""
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1000:  # upscale small images
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return pytesseract.image_to_string(img).strip()


def extract_page_images_text(page) -> list[str]:
    """OCR text of every image on the page."""
    texts = []
    for image in page.images:
        try:
            t = ocr_image(image.data)
        except Exception as e:  # one broken image must not stop the pipeline
            print(f"  ! OCR failed for {image.name}: {e}")
            continue
        if t:
            texts.append(t)
    return texts


def extract_pdf(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    source = Path(pdf_path).name
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pdf_text = extract_pdf_text(page)
        ocr_texts = extract_page_images_text(page)

        # Combine (page-wise): text layer first, then each image's OCR text
        parts = [pdf_text] if pdf_text else []
        parts += [f"[Image text]\n{t}" for t in ocr_texts]
        combined = "\n\n".join(parts)

        pages.append({
            "source": source,
            "page": i,
            "pdf_text": pdf_text,
            "ocr_texts": ocr_texts,
            "combined_text": combined,
        })
        print(f"Page {i:>2}: pdf_text={len(pdf_text):>4} chars | "
              f"images_with_text={len(ocr_texts)} | combined={len(combined):>4} chars")
    return pages


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "rag_test_10pages.pdf"
    pages = extract_pdf(pdf)
    out = Path("output")
    out.mkdir(exist_ok=True)
    (out / "extracted_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone: {len(pages)} pages saved to output/extracted_pages.json")
