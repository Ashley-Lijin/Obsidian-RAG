import networkx as nx
from sentence_transformers import CrossEncoder
from src.retrieval.embedder import get_embedder

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve(query: str, notes: dict, collection, graph, top_k: int = 4, depth: int = 1, max_expanded: int = 5, distance_threshold: float = 0.5) -> dict:
    client, collection_name = collection
    query_vector = get_embedder().encode(query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k * 3,
        with_payload=True,
    ).points

    # Qdrant returns cosine similarity scores; convert to distance (0=identical, 2=opposite)
    chunks = []
    for hit in results:
        dist = 1.0 - hit.score
        if dist <= distance_threshold:
            chunks.append({"text": hit.payload["text"], "note_name": hit.payload["note_name"], "distance": dist})

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
        "context": context,
    }
