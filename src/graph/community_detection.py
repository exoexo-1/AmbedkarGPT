# src/graph/community_detection.py

import pickle
import json
from pathlib import Path
import networkx as nx
import community as community_louvain
from collections import defaultdict, Counter

# ---------- CONFIG ----------
GRAPH_FILES = {
    "sparse": Path("data/graph/knowledge_graph.gpickle"),
    "dense": Path("data/graph/knowledge_graph_dense.gpickle")
}

OUT_DIR = Path("data/graph")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_community_detection(graph_name, graph_path):
    print(f"\n=== Running community detection on {graph_name} graph ===")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Louvain expects undirected graph
    G_undirected = G.to_undirected()

    print("Nodes:", G_undirected.number_of_nodes())
    print("Edges:", G_undirected.number_of_edges())

    # Run Louvain
    partition = community_louvain.best_partition(
        G_undirected,
        weight="weight",
        resolution=1.0,
        random_state=42
    )

    # Attach community ID to nodes
    nx.set_node_attributes(G_undirected, partition, "community")

    # Analyze communities
    community_nodes = defaultdict(list)
    for node, cid in partition.items():
        community_nodes[cid].append(node)

    stats = {
        "graph": graph_name,
        "num_nodes": G_undirected.number_of_nodes(),
        "num_edges": G_undirected.number_of_edges(),
        "num_communities": len(community_nodes),
        "largest_community_size": max(len(v) for v in community_nodes.values()),
        "smallest_community_size": min(len(v) for v in community_nodes.values()),
    }

    # Build rich community objects (SemRAG-ready)
    community_summaries = {}

    for cid, nodes in community_nodes.items():
        # Sort entities by importance (degree in community subgraph)
        degrees = dict(G_undirected.degree(nodes))
        sorted_entities = [
            n for n, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        ]

        community_summaries[str(cid)] = {
            "size": len(nodes),
            "entities": sorted_entities,   # ✅ already ordered by importance
            "summary": None,               # placeholder for LLM
            "embedding_index": None        # filled after summarization
        }



    # Save results
    out = {
        "stats": stats,
        "communities": community_summaries
    }

    out_path = OUT_DIR / f"communities_{graph_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved →", out_path)
    print("Communities:", stats["num_communities"])
    print("Largest community size:", stats["largest_community_size"])

    return out

if __name__ == "__main__":
    results = {}
    for name, path in GRAPH_FILES.items():
        results[name] = run_community_detection(name, path)
