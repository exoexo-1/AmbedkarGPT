#src/retrieval/global_search.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# ----------------- UTILS -----------------
def cosine_sim(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# ----------------- CONFIG -----------------
COMMUNITIES_JSON = Path("data/graph/communities_dense.json")
COMMUNITY_EMB = Path("data/graph/community_summary_embeddings.npy")
CHUNKS_JSON = Path("data/processed/chunks.json")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_COMMUNITIES = 3
TOP_K_POINTS_PER_COMMUNITY = 3

# ----------------- LOAD -----------------
embedder = SentenceTransformer(EMB_MODEL)

with COMMUNITIES_JSON.open("r", encoding="utf-8") as f:
    communities = json.load(f)["communities"]

community_embeddings = np.load(COMMUNITY_EMB)

with CHUNKS_JSON.open("r", encoding="utf-8") as f:
    chunks = json.load(f)["chunks"]

# ----------------- MAIN -----------------
def global_graph_rag_search(query):
    """
    Implements Equation (5) from SemRAG paper
    (Community summaries + subchunks compete as points)
    """
    q_emb = embedder.encode(query)

    # ---- Step 1: Rank communities ----
    comm_scores = []
    for cid, comm in communities.items():
        emb_idx = comm["embedding_index"]
        sim = cosine_sim(q_emb, community_embeddings[emb_idx])
        comm_scores.append((sim, cid))

    comm_scores.sort(reverse=True)
    top_comms = comm_scores[:TOP_K_COMMUNITIES]

    # ---- Step 2: Retrieve points per community ----
    all_points = []

    for comm_score, cid in top_comms:
        comm = communities[cid]
        comm_entities = set(comm["entities"])

        local_points = []

        # ---- (A) Community summary as a point ----
        summary_text = comm.get("summary", "")
        if summary_text:
            summary_emb = community_embeddings[comm["embedding_index"]]
            summary_score = cosine_sim(q_emb, summary_emb)

            local_points.append({
                "score": summary_score,
                "text": f"[COMMUNITY SUMMARY]\n{summary_text}",
                "pages": "Multiple",
                "community_id": cid,
                "type": "community_summary"
            })

        # ---- (B) Subchunks as points ----
        for chunk in chunks:
            for sub in chunk["subchunks"]:
                text = sub["text"]
                if not text:
                    continue

                # entity grounding
                if not any(ent in text for ent in comm_entities):
                    continue

                emb = embedder.encode(text)
                score = cosine_sim(q_emb, emb)

                local_points.append({
                    "score": score,
                    "text": text,
                    "pages": chunk["pages"],
                    "community_id": cid,
                    "type": "subchunk"
                })

        # Top-K points INSIDE this community
        local_points.sort(key=lambda x: x["score"], reverse=True)
        all_points.extend(local_points[:TOP_K_POINTS_PER_COMMUNITY])

    # ---- Step 3: Global ranking ----
    all_points.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "communities": [cid for _, cid in top_comms],
        "results": all_points
    }

# # ----------------- TEST -----------------
# if __name__ == "__main__":
#     print("\n=== GLOBAL GRAPH RAG SEARCH (Eq. 5) ===\n")

#     queries = [
#         "Summarize Ambedkar's views on caste reform",
#         "Explain caste in relation to religion and society"
#     ]

#     for q in queries:
#         out = global_graph_rag_search(q)
#         print(f"\n🌍 QUERY: {q}")
#         print("Top communities:", out["communities"])

#         for i, r in enumerate(out["results"][:15], 1):
#             print(f"\nResult {i}")
#             print("Score:", round(r["score"], 4))
#             print("Community:", r["community_id"])
#             print("Pages:", r["pages"])
#             print(r["text"][:500], "...")
