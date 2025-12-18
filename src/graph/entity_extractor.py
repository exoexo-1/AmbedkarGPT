# src/graph/entity_extractor.py

import json
from pathlib import Path
import spacy
from tqdm import tqdm

# ---------- CONFIG ----------
CHUNKS_JSON = Path("data/processed/chunks.json")
OUT_DIR = Path("data/graph")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_JSON = OUT_DIR / "entities.json"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Entity labels we care about (filter noise)
VALID_ENTITY_LABELS = {
    "DATE",
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "EVENT",
    "NORP",
    "LAW",
    "WORK_OF_ART"
}

def extract_entities():
    with CHUNKS_JSON.open("r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]
    entities = []

    print("Extracting entities from subchunks...")

    for chunk in tqdm(chunks):
        chunk_id = chunk["chunk_id"]
        pages = chunk["pages"]

        for sub_idx, sub in enumerate(chunk["subchunks"]):
            text = sub["text"]
            if not text.strip():
                continue

            doc = nlp(text)

            for ent in doc.ents:
                if ent.label_ not in VALID_ENTITY_LABELS:
                    continue

                entity_record = {
                    "entity": ent.text.strip(),
                    "label": ent.label_,
                    "chunk_id": chunk_id,
                    "subchunk_id": sub_idx,
                    "pages": pages
                }
                entities.append(entity_record)

    with ENTITIES_JSON.open("w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(entities)} entities → {ENTITIES_JSON}")

if __name__ == "__main__":
    extract_entities()
