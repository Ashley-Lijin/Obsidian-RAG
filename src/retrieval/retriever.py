import networkx as nx
from sentence_transformers import CrossEncoder
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from src.retrieval.embedder import get_embedder, get_sparse_embedder

_reranker = None

# Cross-encoder scores are logits; 0 is the sigmoid midpoint.
# Anything below this is likely irrelevant.
RERANKER_THRESHOLD = 0.0


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve(query: str, notes: dict, collection, graph, top_k: int = 4, depth: int = 1, max_expanded: int = 5) -> dict:
    client, collection_name = collection

    dense_vector = get_embedder().encode(query).tolist()
    sparse_emb = next(get_sparse_embedder().query_embed(query))
    sparse_vector = SparseVector(indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist())

    groups = client.query_points_groups(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k * 3),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k * 3),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        group_by="note_name",
        limit=top_k * 3,
        group_size=1,
        with_payload=True,
    ).groups

    chunks = [
        {"text": group.hits[0].payload["text"], "note_name": group.hits[0].payload["note_name"]}
        for group in groups
        if group.hits
    ]

    if not chunks:
        return {"seed": [], "expanded": [], "context": []}

    # Re-rank, then filter — but always keep at least the top result
    reranker = _get_reranker()
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk["score"] = float(score)
    chunks.sort(key=lambda x: x["score"], reverse=True)
    chunks = chunks[:1] + [c for c in chunks[1:] if c["score"] > RERANKER_THRESHOLD]

    # Pick seed notes from top-ranked chunks
    seeds = set()
    chunk_map = {}  # note_name -> best matching chunk text
    for chunk in chunks:
        name = chunk["note_name"]
        if name not in seeds:
            seeds.add(name)
            chunk_map[name] = chunk["text"]
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

    # Seeds: use the matched chunk. Expanded: use full note content for graph context.
    context = []
    for name in seeds:
        if name in notes:
            context.append({"note_name": name, "content": chunk_map[name]})
    for name in expanded:
        if name in notes:
            context.append({"note_name": name, "content": notes[name]["content"]})

    return {
        "seed": list(seeds),
        "expanded": list(expanded),
        "context": context,
    }
