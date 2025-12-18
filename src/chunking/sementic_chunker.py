# semantic_chunker.py
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from math import ceil

from transformers import AutoTokenizer
import numpy.linalg as la

# ---------- CONFIG ----------
MERGED_JSON = Path("data/merged/merged_sentences.json")
EMBED_NPY = Path("data/merged/embeddings.npy")

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_JSON = OUT_DIR / "chunks.json"
CHUNK_EMB_NPY = OUT_DIR / "chunks_embeddings.npy"

# Semantic chunking params (tuneable)
THETA_SIM = 0.86       # similarity threshold (cosine similarity)
TMAX = 1024                 # max tokens per chunk (assignment)
SUB_CHUNK_SIZE = 128        # token size for subchunks
SUB_OVERLAP = 128           # token overlap for subchunks

# Tokenizer (used to count tokens and reconstruct subchunk text)
TOKENIZER_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
    model_max_length=10_000_000,
    truncation=False
)



# ---------- helpers ----------
def load_merged(merged_json_path: Path) -> List[Dict[str, Any]]:
    with merged_json_path.open("r", encoding="utf-8") as fh:
        j = json.load(fh)
    merged = j["merged"]
    return merged, j.get("meta", {})

def load_embeddings(emb_path: Path) -> np.ndarray:
    return np.load(str(emb_path))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = la.norm(a)
    nb = la.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def group_adjacent_by_threshold(embeddings: np.ndarray, theta: float) -> List[List[int]]:
    """
    Given embeddings of merged windows (shape [N, D]),
    compute cosine similarity between adjacent windows and group indices into chunks:
    - start at idx 0
    - while sim(i, i+1) >= theta => keep same chunk
    - else boundary
    Returns list of chunk index lists, e.g. [[0,1,2], [3,4], ...]
    """
    N = embeddings.shape[0]
    if N == 0:
        return []
    chunks = []
    current = [0]
    for i in range(N - 1):
        s = cosine_sim(embeddings[i], embeddings[i+1])
        if s >= theta:
            current.append(i+1)
        else:
            chunks.append(current)
            current = [i+1]
    if current:
        chunks.append(current)
    return chunks

def tokens_from_text(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)

def text_from_token_ids(token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)

def split_tokens_into_subchunks(token_ids: List[int], size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Split list of token ids into windows of length <= size with given overlap (in tokens).
    Returns list of dicts: { 'token_start', 'token_end', 'token_ids', 'text', 'num_tokens' }
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        # If overlap equals size, step will be zero; we force overlap < size by reducing overlap
        overlap = int(size * 0.5)
    step = size - overlap
    out = []
    total = len(token_ids)
    i = 0
    while i < total:
        j = min(i + size, total)
        window = token_ids[i:j]
        out.append({
            "token_start": i,
            "token_end": j,
            "token_ids": window,
            "text": text_from_token_ids(window),
            "num_tokens": len(window)
        })
        if j == total:
            break
        i += step
    return out

def mean_embedding_of_indices(embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros((embeddings.shape[1],), dtype=np.float32)
    return np.mean(embeddings[indices, :], axis=0)

# ---------- main pipeline ----------
def build_chunks(merged_json=MERGED_JSON,
                 embed_npy=EMBED_NPY,
                 theta=THETA_SIM,
                 Tmax=TMAX,
                 sub_size=SUB_CHUNK_SIZE,
                 sub_overlap=SUB_OVERLAP):
    print("Loading merged windows metadata:", merged_json)
    merged, meta = load_merged(merged_json)
    print("Loading embeddings:", embed_npy)
    emb = load_embeddings(embed_npy)
    assert emb.shape[0] == len(merged), "Embeddings count and merged windows count mismatch"

    print(f"Total merged windows: {len(merged)}; embedding dim: {emb.shape[1]}")
    print(f"Grouping adjacent merged windows with theta={theta} ...")
    groups = group_adjacent_by_threshold(emb, theta)
    print(f"Produced {len(groups)} semantic chunks (pre-token-split).")

    chunks_out = []
    chunk_embeddings = []
    for cid, group_inds in enumerate(tqdm(groups, desc="chunks")):
        # Build full chunk text = concatenation of merged windows in order
        texts = [ merged[idx]["text"] for idx in group_inds ]
        chunk_text = " ".join(texts).strip()
        pages = sorted({ p for idx in group_inds for p in merged[idx]["pages"] })
        num_tokens = len(tokens_from_text(chunk_text))
        # compute chunk embedding as mean of member merged embeddings
        chunk_emb = mean_embedding_of_indices(emb, group_inds).astype(np.float32)
        subchunks = []
        if num_tokens <= Tmax:
            # single chunk
            subchunks = [{
                "text": chunk_text,
                "tokens": num_tokens,
                "token_ids": tokens_from_text(chunk_text),
                "token_start": 0,
                "token_end": num_tokens
            }]
        else:
            # split into token-level overlapping subchunks
            token_ids = tokens_from_text(chunk_text)
            windows = split_tokens_into_subchunks(token_ids, size=sub_size, overlap=sub_overlap)
            # windows entries already have decoded text and num_tokens
            subchunks = []
            for w in windows:
                subchunks.append({
                    "text": w["text"],
                    "tokens": w["num_tokens"],
                    "token_start": w["token_start"],
                    "token_end": w["token_end"]
                })

        is_container = num_tokens > Tmax

        chunks_out.append({
            "chunk_id": cid,
            "merged_ids": group_inds,
            "pages": pages,
            "text": chunk_text if not is_container else None,
            "tokens": num_tokens,
            "subchunks": subchunks,
            "is_container": is_container,   # ⭐ IMPORTANT FIX
            "emb_index": cid
        })

        chunk_embeddings.append(chunk_emb)

    # Save chunk embeddings array
    chunk_embeddings = np.vstack(chunk_embeddings) if len(chunk_embeddings) > 0 else np.zeros((0, emb.shape[1]), dtype=np.float32)
    np.save(CHUNK_EMB_NPY, chunk_embeddings)
    print(f"Saved chunk embeddings -> {CHUNK_EMB_NPY} (shape {chunk_embeddings.shape})")

    # Remove token_ids from JSON export to keep it small (we reconstructed text already)
    # and ensure JSON-serializable embeddings (we saved separately)
    for ch in chunks_out:
        for sc in ch["subchunks"]:
            # subchunk text exists; keep it. Do not include token arrays.
            if "token_ids" in sc:
                sc.pop("token_ids", None)
    # Save chunks JSON (without numeric embeddings)
    with CHUNKS_JSON.open("w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "theta_sim": theta,
                "Tmax": Tmax,
                "sub_chunk_size": sub_size,
                "sub_overlap": sub_overlap,
                "tokenizer": TOKENIZER_NAME,
                "n_merged": len(merged),
                "n_chunks": len(chunks_out)
            },
            "chunks": chunks_out
        }, fh, indent=2, ensure_ascii=False)

    print(f"Saved chunks metadata -> {CHUNKS_JSON}")
    print("Done.")
    return CHUNKS_JSON, CHUNK_EMB_NPY

if __name__ == "__main__":
    build_chunks()
