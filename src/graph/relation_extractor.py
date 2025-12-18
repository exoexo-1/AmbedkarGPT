# src/graph/relation_extractor.py

import json
from pathlib import Path
import spacy
from tqdm import tqdm

# ---------- CONFIG ----------
CHUNKS_JSON = Path("data/processed/chunks.json")
ENTITIES_JSON = Path("data/graph/entities.json")

OUT_DIR = Path("data/graph")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RELATIONS_JSON = OUT_DIR / "relations.json"

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Dependency labels we consider meaningful
SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "pobj", "attr"}

def load_entities_by_location(entities):
    """
    Index entities by (chunk_id, subchunk_id)
    """
    ent_map = {}
    for e in entities:
        key = (e["chunk_id"], e["subchunk_id"])
        ent_map.setdefault(key, set()).add(e["entity"])
    return ent_map

def extract_relations():
    # Load data
    with CHUNKS_JSON.open("r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    with ENTITIES_JSON.open("r", encoding="utf-8") as f:
        entities = json.load(f)

    entity_lookup = load_entities_by_location(entities)
    relations = []

    print("Extracting relations from subchunks...")

    for chunk in tqdm(chunks):
        chunk_id = chunk["chunk_id"]
        pages = chunk["pages"]

        for sub_idx, sub in enumerate(chunk["subchunks"]):
            key = (chunk_id, sub_idx)
            if key not in entity_lookup:
                continue

            text = sub["text"]
            if not text.strip():
                continue

            doc = nlp(text)
            known_entities = entity_lookup[key]

            # Deduplicate relations per subchunk
            seen_triples = set()

            for token in doc:
                if token.dep_ not in SUBJECT_DEPS:
                    continue

                subj = token.text
                head = token.head  # verb

                # Normalize relation
                relation = head.lemma_.strip().lower()
                if not relation.isalpha():
                    continue

                for child in head.children:
                    if child.dep_ not in OBJECT_DEPS:
                        continue

                    obj = child.text

                    # Entity grounding
                    if subj not in known_entities or obj not in known_entities:
                        continue

                    # ❌ Remove self-loops
                    if subj == obj:
                        continue

                    triple = (subj, relation, obj)
                    if triple in seen_triples:
                        continue

                    seen_triples.add(triple)

                    relations.append({
                        "subject": subj,
                        "relation": relation,
                        "object": obj,
                        "chunk_id": chunk_id,
                        "subchunk_id": sub_idx,
                        "pages": pages
                    })

    with RELATIONS_JSON.open("w", encoding="utf-8") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(relations)} relations → {RELATIONS_JSON}")

if __name__ == "__main__":
    extract_relations()
