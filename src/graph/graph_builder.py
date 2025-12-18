import json
import pickle
import itertools
from pathlib import Path
from collections import defaultdict

import networkx as nx
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("data/graph")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_JSON = DATA_DIR / "entities.json"
RELATIONS_JSON = DATA_DIR / "relations.json"

BASE_GRAPH_PATH = DATA_DIR / "knowledge_graph.gpickle"
DENSE_GRAPH_PATH = DATA_DIR / "knowledge_graph_dense.gpickle"
STATS_PATH = DATA_DIR / "graph_stats.json"


# ============================================================
# STEP 1: BUILD BASE KNOWLEDGE GRAPH (Entities + Relations)
# ============================================================
def build_base_graph():
    print("\n[1/3] Building base knowledge graph...")

    with ENTITIES_JSON.open("r", encoding="utf-8") as f:
        entities = json.load(f)

    with RELATIONS_JSON.open("r", encoding="utf-8") as f:
        relations = json.load(f)

    G = nx.DiGraph()

    # ---------- Add Nodes ----------
    entity_meta = defaultdict(lambda: {
        "labels": set(),
        "pages": set(),
        "chunks": set()
    })

    for e in entities:
        name = e["entity"]
        entity_meta[name]["labels"].add(e["label"])
        entity_meta[name]["pages"].update(e["pages"])
        entity_meta[name]["chunks"].add(e["chunk_id"])

    for entity, meta in entity_meta.items():
        G.add_node(
            entity,
            labels=list(meta["labels"]),
            pages=sorted(meta["pages"]),
            chunks=sorted(meta["chunks"])
        )

    # ---------- Add Edges (Relations) ----------
    edge_counter = defaultdict(int)
    edge_meta = defaultdict(lambda: {
        "pages": set(),
        "chunks": set()
    })

    for r in relations:
        key = (r["subject"], r["relation"], r["object"])
        edge_counter[key] += 1
        edge_meta[key]["pages"].update(r["pages"])
        edge_meta[key]["chunks"].add(r["chunk_id"])

    for (subj, rel, obj), weight in edge_counter.items():
        if subj not in G or obj not in G:
            continue

        G.add_edge(
            subj,
            obj,
            relation=rel,
            weight=weight,
            pages=sorted(edge_meta[(subj, rel, obj)]["pages"]),
            chunks=sorted(edge_meta[(subj, rel, obj)]["chunks"])
        )

    with open(BASE_GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✔ Base graph saved →", BASE_GRAPH_PATH)
    print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())

    return G, entities, relations


# ============================================================
# STEP 2: ADD CO-OCCURRENCE EDGES (DENSIFICATION)
# ============================================================
def add_cooccurrence_edges(G, entities):
    print("\n[2/3] Adding co-occurrence edges (dense graph)...")

    # Group entities by chunk
    chunk_entities = defaultdict(set)
    for e in entities:
        chunk_entities[e["chunk_id"]].add(e["entity"])

    cooc_counter = defaultdict(int)

    for ents in tqdm(chunk_entities.values()):
        if len(ents) < 2:
            continue
        for a, b in itertools.combinations(sorted(ents), 2):
            cooc_counter[(a, b)] += 1

    # Add co-occurrence edges
    for (a, b), weight in cooc_counter.items():
        if not G.has_node(a) or not G.has_node(b):
            continue

        if G.has_edge(a, b):
            G[a][b]["weight"] += weight
            G[a][b]["relation"] += "|CO_OCCURS"
        else:
            G.add_edge(
                a,
                b,
                relation="CO_OCCURS",
                weight=weight
            )

    with open(DENSE_GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✔ Dense graph saved →", DENSE_GRAPH_PATH)
    print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())

    return G


# ============================================================
# STEP 3: GRAPH STATISTICS & ANALYSIS
# ============================================================
def analyze_and_save_stats(G, entities, relations):
    print("\n[3/3] Computing graph statistics...")

    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

    top_hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_entities_raw": len(entities),
        "num_relations_raw": len(relations),
        "graph_density": nx.density(G),
        "average_degree": round(avg_degree, 3),
        "top_10_hubs": [{ "node": n, "degree": d } for n, d in top_hubs]
    }

    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✔ Stats saved →", STATS_PATH)
    print("Average degree:", round(avg_degree, 2))
    print("Top hubs:")
    for n, d in top_hubs:
        print(" ", n, "→", d)


# ============================================================
# MAIN PIPELINE
# ============================================================
def build_full_graph():
    G, entities, relations = build_base_graph()
    G = add_cooccurrence_edges(G, entities)
    analyze_and_save_stats(G, entities, relations)

    print("\n✅ Knowledge Graph construction completed successfully.")


if __name__ == "__main__":
    build_full_graph()
