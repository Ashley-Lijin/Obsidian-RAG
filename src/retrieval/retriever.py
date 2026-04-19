import networkx as nx
from sentence_transformers import CrossEncoder

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _best_chunk_for_note(note_name: str, query: str, collection) -> str | None:
    """Query ChromaDB for the single best chunk of a given note."""
    results = collection.query(
        query_texts=[query],
        n_results=1,
        where={"note_name": note_name},
        include=["documents"]
    )
    docs = results.get("documents", [[]])
    return docs[0][0] if docs and docs[0] else None


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

    # Pick seed notes from top-ranked chunks (best chunk per note)
    seeds = set()
    best_chunk: dict[str, str] = {}
    for chunk in chunks:
        name = chunk["note_name"]
        if name not in best_chunk:
            seeds.add(name)
            best_chunk[name] = chunk["text"]
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

    # Fetch best chunk for expanded notes not already in results
    for name in expanded:
        if name not in best_chunk:
            chunk_text = _best_chunk_for_note(name, query, collection)
            best_chunk[name] = chunk_text if chunk_text else notes[name]["content"][:500]

    context = [
        {"note_name": name, "content": best_chunk[name]}
        for name in seeds | expanded
        if name in best_chunk
    ]

    return {
        "seed": list(seeds),
        "expanded": list(expanded),
        "context": context
    }
