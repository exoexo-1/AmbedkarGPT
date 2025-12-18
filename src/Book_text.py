# extract_pdf.py
import os
import json
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict

# Optional OCR tools
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

PDF_PATH = Path("data/Ambedkar_book.pdf")   # ensure this path is correct
OUT_DIR = Path("data/pages")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_TXT = Path("data/raw_text.txt")
INDEX_JSON = OUT_DIR / "pages_index.json"

def extract_with_pypdf(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = ""
        pages_text.append(text)
    return pages_text

def extract_with_ocr(pdf_path: Path, dpi=300) -> List[str]:
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR dependencies not available (pdf2image/pytesseract).")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    pages_text = []
    for img in images:
        # convert to grayscale (optional)
        text = pytesseract.image_to_string(img)
        pages_text.append(text)
    return pages_text

def write_outputs(pages_text: List[str], pdf_path: Path):
    # Combined raw text
    combined = "\n\n".join([f"[PAGE {i+1}]\n{p}" for i, p in enumerate(pages_text)])
    RAW_TXT.write_text(combined, encoding="utf-8")

    # Per-page files and index
    index = []
    for i, p in enumerate(pages_text):
        page_num = i + 1
        page_file = OUT_DIR / f"page_{page_num:03d}.txt"
        page_file.write_text(p, encoding="utf-8")
        index.append({
            "page": page_num,
            "file": str(page_file),
            "num_chars": len(p),
            "num_words": len(p.split())
        })
    INDEX_JSON.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved combined text -> {RAW_TXT}")
    print(f"Saved {len(pages_text)} page files under -> {OUT_DIR}")
    print(f"Saved index -> {INDEX_JSON}")

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Please place Ambedkar_book.pdf there.")

    # 1) Try pypdf extraction
    print("Extracting text using pypdf...")
    pages_text = extract_with_pypdf(PDF_PATH)

    # 2) Check how many pages are empty -> if many empty, attempt OCR fallback
    empty_pages = sum(1 for t in pages_text if not t or t.strip() == "")
    total_pages = len(pages_text)
    print(f"pypdf extraction: {total_pages} pages, {empty_pages} empty pages")

    # Heuristic: if >30% pages are empty -> try OCR fallback (if available)
    if empty_pages / max(1, total_pages) > 0.30:
        if OCR_AVAILABLE:
            print("Many empty pages detected. Running OCR fallback (pdf2image + pytesseract). This will be slower...")
            ocr_text = extract_with_ocr(PDF_PATH)
            # if OCR returned non-empty for most pages, use it
            ocr_empty = sum(1 for t in ocr_text if not t or t.strip() == "")
            print(f"OCR extraction: {len(ocr_text)} pages, {ocr_empty} empty")
            # choose OCR if it produced fewer empties
            if ocr_empty < empty_pages:
                pages_text = ocr_text
                print("Using OCR output.")
            else:
                print("Keeping pypdf output (OCR did not improve).")
        else:
            print("OCR fallback not available (pdf2image/pytesseract not installed). Keeping pypdf output.")
    else:
        print("pypdf extraction looks good; using it.")

    # 3) Basic cleanup: normalize consecutive whitespace
    cleaned = [ " ".join(p.split()) for p in pages_text ]
    write_outputs(cleaned, PDF_PATH)

if __name__ == "__main__":
    main()
