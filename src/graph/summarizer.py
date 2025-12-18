import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
COMMUNITIES_JSON = Path("data/graph/communities_dense.json")
ENTITIES_JSON = Path("data/graph/entities.json")
CHUNKS_JSON = Path("data/processed/chunks.json")
CHUNK_EMB = Path("data/processed/chunks_embeddings.npy")

OUT_EMB_NPY = Path("data/graph/community_summary_embeddings.npy")

LLM_MODEL = "gpt-4o-mini"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_ENTITIES = 20
MAX_SUBCHUNKS_PER_ENTITY = 1
MAX_TOTAL_SUBCHUNKS = 20
MAX_CHARS_PER_SUBCHUNK = 800

# ---------- CLIENTS ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer(EMB_MODEL)

# ---------- HELPERS ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def build_prompt(entities, passages):
    prompt = (
        "You are an expert academic analyst.\n\n"
        "You are given a group of entities extracted from a scholarly text )"
        "and representative text passages related to them (chunks).\n\n"
        "TASK:\n"
        "Write a concise but information-dense community summary that:\n"
        "1. Explains the central theme of this group\n"
        "2. Mentions the most important entities and concepts\n"
        "3. Captures the relationships, story or arguments connecting them\n"
        "4. Is suitable for semantic retrieval (not narrative writing)\n\n"
        "Do NOT add new facts or opinions.\n"
        "Keep the summary factual and compact (150–250 words).\n\n"
        "ENTITIES:\n"
        f"{', '.join(entities)}\n\n"
        "TEXT PASSAGES:\n"
    )

    for p in passages:
        prompt += f"- {p}\n\n"

    return prompt

def call_llm(prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ---------- MAIN ----------
def generate_community_summaries():
    communities = load_json(COMMUNITIES_JSON)
    entities = load_json(ENTITIES_JSON)
    chunks = load_json(CHUNKS_JSON)["chunks"]
    chunk_embeddings = np.load(CHUNK_EMB)

    # Maps
    chunk_map = {c["chunk_id"]: c for c in chunks}

    entity_chunks = defaultdict(set)
    for e in entities:
        entity_chunks[e["entity"]].add(e["chunk_id"])

    summary_embeddings = []
    emb_index = 0

    print("Generating / embedding community summaries...")

    for cid, comm in tqdm(communities["communities"].items()):
        # ---------- SUMMARY ----------
        if not comm.get("summary"):
            top_entities = comm["entities"][:TOP_ENTITIES]

            member_chunk_ids = set()
            for ent in top_entities:
                member_chunk_ids |= entity_chunks.get(ent, set())

            if not member_chunk_ids:
                continue

            centroid = np.mean(chunk_embeddings[list(member_chunk_ids)], axis=0)

            candidates = []
            for ent in top_entities:
                for cid_ in entity_chunks.get(ent, []):
                    chunk = chunk_map.get(cid_)
                    if not chunk:
                        continue

                    for sub in chunk["subchunks"]:
                        text = sub["text"].strip()
                        if not text:
                            continue

                        emb = embedder.encode(text)
                        score = cosine_sim(emb, centroid)
                        candidates.append((score, text[:MAX_CHARS_PER_SUBCHUNK]))

            candidates.sort(reverse=True, key=lambda x: x[0])
            selected = [t for _, t in candidates[:MAX_TOTAL_SUBCHUNKS]]

            prompt = build_prompt(top_entities, selected)
            comm["summary"] = call_llm(prompt)

        # ---------- EMBEDDING ----------
        summary_vec = embedder.encode(comm["summary"])
        summary_embeddings.append(summary_vec)

        comm["embedding_index"] = emb_index
        emb_index += 1

        # 🔴 ENSURE no embedding stored in JSON
        comm.pop("embedding", None)

    # Save embeddings
    summary_embeddings = np.vstack(summary_embeddings)
    np.save(OUT_EMB_NPY, summary_embeddings)

    # Save updated communities JSON
    with open(COMMUNITIES_JSON, "w", encoding="utf-8") as f:
        json.dump(communities, f, indent=2, ensure_ascii=False)

    print("✅ DONE")
    print("• communities_dense.json updated")
    print("• community_summary_embeddings.npy saved")
    print("• Embedding shape:", summary_embeddings.shape)

if __name__ == "__main__":
    generate_community_summaries()
