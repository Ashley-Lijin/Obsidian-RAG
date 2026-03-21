import networkx as nx

def retrieve(query: str, notes: dict, collection, graph, top_k: int = 4, depth: int = 1, max_expanded: int = 2, distance_threshold: float = 1.2) -> dict:
    # Over-fetch chunks with distances so we can filter by relevance quality
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,
        include=["metadatas", "distances"]
    )

    # Track the best (lowest) distance per note — lower = more relevant
    best_distance: dict[str, float] = {}
    for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
        name = meta['note_name']
        if name not in best_distance or dist < best_distance[name]:
            best_distance[name] = dist

    # Keep only notes within the relevance threshold, ranked by distance
    seeds = set()
    for name, dist in sorted(best_distance.items(), key=lambda x: x[1]):
        if dist > distance_threshold:
            break
        seeds.add(name)
        if len(seeds) >= top_k:
            break

    expanded = set()

    for seed in seeds:
        if seed in graph:
            undirected = graph.to_undirected()
            nodes = nx.single_source_shortest_path_length(undirected, seed, cutoff=depth)
            expanded.update(nodes.keys())

    expanded = expanded - seeds
    # Cap expanded notes to prevent context overflow in small local models
    expanded = set(list(expanded)[:max_expanded])

    context_notes = seeds.union(expanded)

    context = []
    for note_name in context_notes:
        if note_name in notes:
            context.append(
                {
                    "note_name" : note_name,
                    "content" : notes[note_name]["content"]
                }
            )

    return {
        "seed" : list(seeds),
        "expanded" : list(expanded),
        "context" : context
    }
