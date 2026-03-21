import networkx as nx

def retrieve(query: str, notes: dict, collection, graph, top_k: int = 3, depth: int = 1) -> dict:
    results = collection.query(
    query_texts=[query],
    n_results=top_k
    )


    seeds = set()

    for meta in results['metadatas'][0]:
        seeds.add(meta['note_name'])

    expanded = set()

    for seed in seeds:
        if seed in graph:
            undirected = graph.to_undirected()
            nodes = nx.single_source_shortest_path_length(undirected, seed, cutoff=depth)
            expanded.update(nodes.keys())


    expanded = expanded - seeds

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
