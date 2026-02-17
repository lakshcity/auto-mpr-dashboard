import time
import pickle
import re
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import pytesseract

# Explicitly point to the executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from pdf2image import convert_from_path

# Replace this with the actual path where you extracted the bin folder
POPPLER_PATH = r'C:\poppler\poppler-25.12.0\Library\bin'

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "data" / "pdfs"
INDEX_PATH = BASE_DIR / "data" / "pdf_index.faiss"
META_PATH = BASE_DIR / "data" / "pdf_meta.pkl"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80

# If Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", "")
    return text.strip()


def chunk_text(text, chunk_size=300, overlap=80):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []

    for para in paragraphs:
        words = para.split()

        if len(" ".join(current_chunk).split()) + len(words) <= chunk_size:
            current_chunk.append(para)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words), para]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_text_and_images(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    # Extract regular text
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    # Extract images and apply OCR
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                full_text += "\n" + ocr_text + "\n"
    except Exception as e:
        print(f"OCR failed for {pdf_path.name}: {e}")

    return clean_text(full_text)



def load_and_chunk_pdfs(pdf_dir: Path):
    docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")

        full_text = extract_text_and_images(pdf_file)

        if full_text:
            chunks = chunk_text(full_text)

            for idx, chunk in enumerate(chunks):
                docs.append({
                    "source": pdf_file.name,
                    "chunk_id": idx,
                    "text": chunk,
                    "length": len(chunk.split())
                })

    if not docs:
        raise RuntimeError("No valid PDFs found")

    return docs


def build_pdf_index():
    print("=== PDF INDEX WITH OCR STARTED ===")
    start = time.time()

    docs = load_and_chunk_pdfs(PDF_DIR)
    texts = [d["text"] for d in docs]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("✅ OCR-enabled PDF index built")
    print(f"Total chunks: {len(docs)}")
    print(f"Time: {round(time.time() - start, 2)} sec")


if __name__ == "__main__":
    build_pdf_index()
