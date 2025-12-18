"""
Purpose:
  - Read per-page text files from data/pages (created by your PDF extraction step).
  - Split pages into sentences using spaCy.
  - For each sentence index i, create a "buffer window" consisting of sentences[i-buffer : i+buffer+1].
  - Deduplicate consecutive identical windows (sliding windows cause duplicates).
  - Compute embeddings for each merged window using sentence-transformers (all-MiniLM-L6-v2).
  - Save merged metadata to data/merged/merged_sentences.json and embeddings to data/merged/embeddings.npy.

Outputs:
  - data/merged/merged_sentences.json
  - data/merged/embeddings.npy
"""

from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# NLP / embeddings
import spacy
from sentence_transformers import SentenceTransformer

# ----------------- CONFIG -----------------
PAGES_DIR = Path("data/pages")                    # folder containing page_001.txt ... page_094.txt
OUT_DIR = Path("data/merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_JSON = OUT_DIR / "merged_sentences.json"
EMBED_NPY = OUT_DIR / "embeddings.npy"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"            # recommended model (fast & good)
BUFFER_SIZE = 2                                 # ±b sentences for each merged window (tuneable)
EMBED_BATCH_SIZE = 64                            # batch size for embeddings (tuneable)
SPACY_MODEL = "en_core_web_sm"                   # spaCy model for sentence splitting

# ----------------- Utilities -----------------
def load_page_texts(pages_dir: Path) -> List[Dict[str, Any]]:
    """
    Read page files (page_###.txt) and return ordered list of dicts: { page, file, text }
    """
    page_files = sorted(pages_dir.glob("page_*.txt"))
    pages = []
    for p in page_files:
        txt = p.read_text(encoding="utf-8")
        page_num = int(p.stem.split("_")[-1])
        pages.append({"page": page_num, "file": str(p), "text": txt})
    pages.sort(key=lambda x: x["page"])
    return pages

def split_pages_to_sentences(pages: List[Dict[str, Any]], nlp) -> List[Dict[str, Any]]:
    """
    Convert pages into a list of sentence dicts:
      { sent_id, text, page, char_start (optional), char_end (optional) }
    Note: char offsets here are not used downstream but kept for traceability if needed.
    """
    sentences = []
    sid = 0
    for pg in pages:
        text = pg["text"]
        if not text or text.strip() == "":
            continue
        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            sentences.append({
                "sent_id": sid,
                "text": s,
                "page": pg["page"]
            })
            sid += 1
    return sentences

def make_buffer_merged(sentences: List[Dict[str, Any]], buffer_size: int) -> List[Dict[str, Any]]:
    """
    For each sentence index i, create merged_text of sentences[i-buffer : i+buffer+1].
    Deduplicate consecutive identical merged windows (sliding overlap causes repeated windows).
    Returns merged windows with metadata:
      { merged_id, text, base_sent_ids, pages, num_words }
    """
    texts = [s["text"] for s in sentences]
    n = len(texts)
    merged = []
    prev_text = None
    merged_id = 0
    for i in range(n):
        start = max(0, i - buffer_size)
        end = min(n, i + buffer_size + 1)
        merged_text = " ".join(texts[start:end]).strip()
        if merged_text == prev_text:
            continue  # skip duplicates
        base_sent_ids = list(range(start, end))
        pages = sorted({ sentences[j]["page"] for j in base_sent_ids })
        merged.append({
            "merged_id": merged_id,
            "text": merged_text,
            "base_sent_ids": base_sent_ids,
            "pages": pages,
            "num_words": len(merged_text.split())
        })
        prev_text = merged_text
        merged_id += 1
    return merged

def compute_embeddings_texts(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """
    Compute sentence-transformers embeddings for a list of texts and return a float32 numpy array.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)

# ----------------- Main -----------------
def main():
    print("Loading spaCy model for sentence splitting...")
    nlp = spacy.load(SPACY_MODEL)

    print(f"Reading pages from {PAGES_DIR} ...")
    pages = load_page_texts(PAGES_DIR)
    print(f"Found {len(pages)} page files.")

    print("Splitting pages into sentences (spaCy)...")
    sentences = split_pages_to_sentences(pages, nlp)
    print(f"Extracted {len(sentences)} sentences from pages.")

    print(f"Creating buffer-merged windows with buffer_size = {BUFFER_SIZE} ...")
    merged = make_buffer_merged(sentences, BUFFER_SIZE)
    print(f"Created {len(merged)} merged windows.")

    # Save initial merged JSON without embeddings (lightweight)
    # We'll add emb_index once embeddings are computed.
    for i, m in enumerate(merged):
        m["merged_id"] = int(m["merged_id"])
        m["emb_index"] = None  # placeholder

    tmp_json = MERGED_JSON.with_suffix(".partial.json")
    with tmp_json.open("w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "spacy_model": SPACY_MODEL,
                "embed_model": EMBED_MODEL_NAME,
                "buffer_size": BUFFER_SIZE,
                "n_sentences": len(sentences),
                "n_merged": len(merged)
            },
            "merged": merged
        }, fh, indent=2, ensure_ascii=False)

    print("Computing embeddings for merged windows (sentence-transformers)...")
    texts = [m["text"] for m in merged]
    embeddings = compute_embeddings_texts(texts, EMBED_MODEL_NAME, batch_size=EMBED_BATCH_SIZE)
    print("Embeddings computed. Shape:", embeddings.shape)

    # Save embeddings numpy
    np.save(EMBED_NPY, embeddings)
    print(f"Saved embeddings -> {EMBED_NPY}")

    # update merged entries with emb_index
    for i, m in enumerate(merged):
        m["emb_index"] = int(i)
        # optionally remove very long text or keep as is for traceability

    # Save final merged_json
    with MERGED_JSON.open("w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "spacy_model": SPACY_MODEL,
                "embed_model": EMBED_MODEL_NAME,
                "buffer_size": BUFFER_SIZE,
                "n_sentences": len(sentences),
                "n_merged": len(merged)
            },
            "merged": merged
        }, fh, indent=2, ensure_ascii=False)

    print(f"Saved merged windows metadata -> {MERGED_JSON}")
    print("Done.")

if __name__ == "__main__":
    main()
