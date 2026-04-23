import networkx as nx
from sentence_transformers import CrossEncoder

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve(query: str, notes: dict, collection, graph, top_k: int = 6, depth: int = 1, max_expanded: int = 5, distance_threshold: float = 1.2) -> dict:
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,
        include=["documents", "metadatas", "distances"]
    )

    # Collect chunks within relevance threshold
    chunks = []
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        if dist <= distance_threshold:
            chunks.append({"text": doc, "note_name": meta['note_name'], "distance": dist})

    if not chunks:
        return {"seed": [], "expanded": [], "context": []}

    # Re-rank with cross-encoder
    reranker = _get_reranker()
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk["score"] = float(score)
    chunks.sort(key=lambda x: x["score"], reverse=True)

    # Pick seed notes from top-ranked chunks (deduplicate by note)
    seeds = set()
    for chunk in chunks:
        seeds.add(chunk["note_name"])
        if len(seeds) >= top_k:
            break

    # Graph expand seeds to related notes
    expanded = set()
    for seed in seeds:
        if seed in graph:
            undirected = graph.to_undirected()
            nodes = nx.single_source_shortest_path_length(undirected, seed, cutoff=depth)
            expanded.update(nodes.keys())
    expanded = expanded - seeds
    expanded = set(list(expanded)[:max_expanded])

    context = [
        {"note_name": name, "content": notes[name]["content"]}
        for name in seeds | expanded
        if name in notes
    ]

    return {
        "seed": list(seeds),
        "expanded": list(expanded),
        "context": context
    }
