"""Generate a 10-page test PDF (text + images containing text) for a RAG ingestion pipeline."""
import os
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image as RLImage, PageBreak)

OUT_DIR = "."
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def make_image(name, title, lines, size=(1200, 600), bg="#F4F7FB"):
    """Create an image that contains readable text (so OCR has work to do)."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, size[0] - 10, size[1] - 10], outline="#1F3A5F", width=4)
    d.text((40, 35), title, font=ImageFont.truetype(FONT_B, 44), fill="#1F3A5F")
    y = 120
    f = ImageFont.truetype(FONT, 32)
    for line in lines:
        d.text((50, y), line, font=f, fill="#111111")
        y += 52
    path = os.path.join(IMG_DIR, name)
    img.save(path)
    return path


def make_flow_image(name, title, steps):
    """Boxes-and-arrows diagram with text inside boxes."""
    w = 1200
    h = 200 + 130 * len(steps)
    img = Image.new("RGB", (w, h), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, font=ImageFont.truetype(FONT_B, 42), fill="#1F3A5F")
    f = ImageFont.truetype(FONT_B, 32)
    y = 120
    for i, s in enumerate(steps):
        d.rounded_rectangle([200, y, 1000, y + 80], radius=18,
                            fill="#DCE9F9", outline="#1F3A5F", width=3)
        d.text((230, y + 20), f"{i + 1}. {s}", font=f, fill="#111111")
        if i < len(steps) - 1:
            d.line([600, y + 80, 600, y + 125], fill="#1F3A5F", width=4)
            d.polygon([(585, y + 112), (615, y + 112), (600, y + 128)], fill="#1F3A5F")
        y += 130
    path = os.path.join(IMG_DIR, name)
    img.save(path)
    return path


def make_table_image(name, title, header, rows):
    w, h = 1200, 180 + 70 * (len(rows) + 1)
    img = Image.new("RGB", (w, h), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, font=ImageFont.truetype(FONT_B, 42), fill="#1F3A5F")
    fb, fr = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 30)
    col_x = [50, 450, 850]
    y = 110
    d.rectangle([40, y, w - 40, y + 60], fill="#1F3A5F")
    for x, t in zip(col_x, header):
        d.text((x, y + 12), t, font=fb, fill="#FFFFFF")
    y += 60
    for r in rows:
        d.rectangle([40, y, w - 40, y + 70], outline="#999999", width=2)
        for x, t in zip(col_x, r):
            d.text((x, y + 18), t, font=fr, fill="#111111")
        y += 70
    path = os.path.join(IMG_DIR, name)
    img.save(path)
    return path


# ---------- images (text lives INSIDE these, not in the PDF text layer) ----------
img_arch = make_flow_image("img_rag_pipeline.png", "RAG Pipeline Overview",
    ["Load PDF documents", "Extract text and run OCR on images",
     "Clean text and remove duplicates", "Split into chunks",
     "Create embeddings and store in vector DB"])

img_stats = make_table_image("img_model_table.png", "Embedding Model Comparison",
    ["Model", "Dimensions", "Max Tokens"],
    [["text-embedding-004", "768", "2048"],
     ["gemini-embedding-001", "3072", "2048"],
     ["all-MiniLM-L6-v2", "384", "256"]])

img_notice = make_image("img_notice.png", "IMPORTANT NOTICE",
    ["Scanned pages must be at least 200 DPI for good OCR.",
     "Always remove duplicate paragraphs before embedding.",
     "Keep page numbers in metadata for easy citation.",
     "Contact: support@example.com  |  Phone: 0300-1234567"])

img_chunk = make_image("img_chunking.png", "Chunking Guidelines",
    ["Chunk size: 800 characters",
     "Chunk overlap: 100 characters",
     "Separator order: paragraph, sentence, word",
     "Store metadata: source file, page number, chunk id"])

img_vector = make_table_image("img_vector_db.png", "Vector Database Options",
    ["Database", "Type", "Best For"],
    [["ChromaDB", "Local", "Prototypes"],
     ["FAISS", "Local", "Fast search"],
     ["Pinecone", "Cloud", "Production"]])

# ---------- PDF ----------
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor="#1F3A5F", spaceAfter=10)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11.5, leading=17, spaceAfter=8)
CAP = ParagraphStyle("Cap", parent=styles["BodyText"], fontSize=9.5, textColor="#555555")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawString(2 * cm, 1.2 * cm, "RAG Test Document - Confidential Sample")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def img_flowable(path, width_cm=15):
    im = Image.open(path)
    ratio = im.size[1] / im.size[0]
    return RLImage(path, width=width_cm * cm, height=width_cm * cm * ratio)


P = lambda t: Paragraph(t, BODY)
story = []

# Page 1 - text only
story += [Paragraph("Introduction to Retrieval-Augmented Generation", H1),
    P("Retrieval-Augmented Generation (RAG) is a technique that combines a large language model with an external knowledge source. Instead of relying only on what the model memorised during training, a RAG system first retrieves relevant passages from a document collection and then uses them as context to generate an answer."),
    P("RAG reduces hallucinations because the answer is grounded in real documents. It also makes it easy to update knowledge: you only need to add new documents to the database, and there is no need to retrain the model."),
    P("A typical RAG application has two phases. The first phase is ingestion, where documents are loaded, cleaned, split into chunks, converted into embeddings and stored in a vector database. The second phase is retrieval, where the user question is embedded and the most similar chunks are fetched and passed to the language model."),
    PageBreak()]

# Page 2 - text + image
story += [Paragraph("The Ingestion Pipeline", H1),
    P("Ingestion is the foundation of every RAG chatbot. If the data entering the database is messy, the answers coming out will be messy too. The diagram below shows the full flow that we will follow in this project."),
    img_flowable(img_arch, 12),
    Paragraph("Figure 1: Overall RAG pipeline (text inside this figure is only available through OCR).", CAP),
    P("Each step has a clear responsibility. Extraction gets raw content from the file, cleaning normalises it, chunking prepares it for search and embedding turns text into numbers that can be compared."),
    PageBreak()]

# Page 3 - text only
story += [Paragraph("Extracting Text from PDF Files", H1),
    P("Many PDF files store their text in a text layer. Python libraries such as pypdf can read this layer page by page and return the text as a string. This is fast and accurate for digitally created documents."),
    P("However, pypdf cannot read text that is part of an image. Scanned pages, screenshots, charts and infographics are stored as pictures, so the extractor will return nothing for them. That is why we need OCR as a second extraction method."),
    P("When extracting, keep track of the page number for every piece of text. This metadata will later allow the chatbot to tell the user exactly which page an answer came from."),
    P("Some PDFs contain headers and footers that repeat on every page. These repeated lines add noise and duplicates to the data and should be removed during the cleaning stage."),
    PageBreak()]

# Page 4 - text + image
story += [Paragraph("Embedding Models", H1),
    P("An embedding model converts a piece of text into a vector of numbers. Texts with similar meaning have vectors that are close to each other. Google provides several embedding models through the Gemini API, and the table below compares a few popular choices."),
    img_flowable(img_stats, 14),
    Paragraph("Figure 2: Embedding model comparison table.", CAP),
    P("Higher dimensions can capture more detail, but they also need more storage and make search slightly slower. For a first project, a 768-dimension model is a good balance."),
    PageBreak()]

# Page 5 - image-heavy
story += [Paragraph("Practical Tips", H1),
    P("The notice below was added as an image on purpose. It simulates a scanned page or a screenshot that a normal text extractor cannot read."),
    img_flowable(img_notice, 15),
    Paragraph("Figure 3: Practical tips (image only).", CAP),
    P("Always remove duplicate paragraphs before embedding."),
    PageBreak()]

# Page 6 - text only
story += [Paragraph("Optical Character Recognition (OCR)", H1),
    P("OCR is the process of converting images of text into machine-readable text. Tesseract is a free and widely used OCR engine, and it can be called from Python using the pytesseract package."),
    P("For best results, images should be clear, high resolution and have good contrast. Converting the image to grayscale and increasing its size before OCR often improves accuracy."),
    P("OCR is never perfect. It may confuse similar characters such as the letter O and the number 0. This is one more reason why a cleaning step with a language model is valuable: it can fix obvious mistakes and restore proper formatting."),
    P("Retrieval-Augmented Generation (RAG) is a technique that combines a large language model with an external knowledge source."),
    PageBreak()]

# Page 7 - text + image
story += [Paragraph("Chunking the Text", H1),
    P("Language models and embedding models can only process a limited amount of text at once, and small focused chunks give more precise search results. Therefore, long documents must be split into chunks before embedding."),
    img_flowable(img_chunk, 14),
    Paragraph("Figure 4: Recommended chunking settings.", CAP),
    P("Overlap between chunks keeps sentences that fall on a boundary from losing their context."),
    PageBreak()]

# Page 8 - text only
story += [Paragraph("Cleaning and Deduplication", H1),
    P("After extraction, the combined text from pypdf and OCR usually contains problems: extra spaces, broken lines, repeated headers and footers, and sometimes the same information appearing twice because it exists both in the text layer and in an image."),
    P("A simple rule-based pass can remove extra whitespace and exact duplicates. A language model can then handle the harder cases, such as near-duplicate sentences, OCR mistakes and inconsistent formatting."),
    P("Important: when using an LLM for cleaning, instruct it never to add new facts and never to summarise. Its job is only to tidy the text, not to change its meaning."),
    P("An embedding model converts a piece of text into a vector of numbers."),
    PageBreak()]

# Page 9 - text + image
story += [Paragraph("Storing Vectors", H1),
    P("Once embeddings are created they must be stored together with the original chunk text and its metadata. A vector database is designed for this job and can quickly find the nearest vectors to a query."),
    img_flowable(img_vector, 14),
    Paragraph("Figure 5: Common vector database options.", CAP),
    P("For learning and prototyping, a local database such as ChromaDB is the easiest to start with because it needs no account and no server."),
    PageBreak()]

# Page 10 - text only (summary)
story += [Paragraph("Summary and Next Steps", H1),
    P("In this document we covered the full ingestion flow of a RAG chatbot: extract text with pypdf, read image text with OCR, combine and clean the data, create embeddings with a Google embedding model and save everything in a vector database."),
    P("Once ingestion works correctly, the next phase is retrieval. The user question is embedded, the most relevant chunks are fetched from the database, and the language model produces a final answer using those chunks as context."),
    P("Good luck with your project! For questions, contact support@example.com."),
    ]

doc = SimpleDocTemplate(os.path.join(OUT_DIR, "rag_test_10pages.pdf"), pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="RAG Test Document", author="Test")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF created")
