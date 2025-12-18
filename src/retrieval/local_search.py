
#src/retrieval/local_search.py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# ----------------- UTILS -----------------
def cosine_sim(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# ----------------- CONFIG -----------------
ENTITIES_JSON = Path("data/graph/entities.json")
CHUNKS_JSON = Path("data/processed/chunks.json")
CHUNK_EMB = Path("data/processed/chunks_embeddings.npy")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TAU_E = 0.35        # query ↔ entity
TAU_D = 0.45        # entity ↔ chunk
TOP_K_ENTITIES = 40
TOP_K_CHUNKS = 5

# ----------------- LOAD -----------------
embedder = SentenceTransformer(EMB_MODEL)

with ENTITIES_JSON.open("r", encoding="utf-8") as f:
    entities = json.load(f)

with CHUNKS_JSON.open("r", encoding="utf-8") as f:
    chunks = json.load(f)["chunks"]

chunk_embeddings = np.load(CHUNK_EMB)

# entity → chunk_ids (graph grounded)
entity_to_chunks = defaultdict(set)
for e in entities:
    entity_to_chunks[e["entity"]].add(e["chunk_id"])

# ----------------- MAIN -----------------
def local_graph_rag_search(query, history=None):
    """
    Implements Equation (4) from SemRAG paper
    """
    q_text = query if history is None else query + " " + history
    q_emb = embedder.encode(q_text)

    # ---- Step 1: Rank entities by similarity ----
    entity_scores = []
    for ent in entity_to_chunks:
        e_emb = embedder.encode(ent)
        sim = cosine_sim(q_emb, e_emb)
        if sim >= TAU_E:
            entity_scores.append((sim, ent))

    entity_scores.sort(reverse=True)
    top_entities = [e for _, e in entity_scores[:TOP_K_ENTITIES]]

    # ---- Step 2: Collect candidate chunks ----
    candidate_chunks = set()
    for ent in top_entities:
        candidate_chunks |= entity_to_chunks[ent]

    # ---- Step 3: Rank chunks ----
    scored_chunks = []
    for cid in candidate_chunks:
        sim = cosine_sim(q_emb, chunk_embeddings[cid])
        if sim >= TAU_D:
            scored_chunks.append((sim, cid))

    scored_chunks.sort(reverse=True)
    top_chunks = scored_chunks[:TOP_K_CHUNKS]

    # ---- Step 4: Return subchunks (points) ----
    results = []
    for score, cid in top_chunks:
        chunk = chunks[cid]
        for sub in chunk["subchunks"]:
            results.append({
                "score": score,
                "text": sub["text"],
                "pages": chunk["pages"],
                "chunk_id": cid
            })

    return {
        "query": query,
        "entities": top_entities,
        "results": results
    }

# # ----------------- TEST -----------------
# if __name__ == "__main__":
#     print("\n=== LOCAL GRAPH RAG SEARCH (Eq. 4) ===\n")

#     queries = [
#         "What is Ambedkar's critique of the caste system?",
#         "What role do Shastras play in Hindu society?"
#     ]

#     for q in queries:
#         out = local_graph_rag_search(q)
#         print(f"\n🔎 QUERY: {q}")
#         print("Top entities:", out["entities"][:50])

#         for i, r in enumerate(out["results"][:5], 1):
#             print(f"\nResult {i}")
#             print("Score:", round(r["score"], 4))
#             print("Pages:", r["pages"])
#             print(r["text"][:300], "...")
