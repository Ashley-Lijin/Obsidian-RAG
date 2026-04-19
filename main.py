from src.ingestion.vault_reader import load_notes
from src.retrieval.embedder import embed_vault, get_collection
import os
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve
from src.retrieval.graph_builder import build_graph
from src.agent.agent import ask, rewrite_query, manage_memory, is_context_sufficient, generate_subqueries

load_dotenv()

notes = load_notes(os.getenv("VAULT_PATH"))
collection = get_collection()
embed_vault(notes, collection)
graph = build_graph(notes)

print("Obsidian RAG — ask anything about your notes (type 'exit' to quit)\n")

MAX_RETRIEVAL_ROUNDS = 5

history = []
long_term_summary = ""
while True:
    query = input("> ")
    if query.strip() == "exit":
        break

    history, long_term_summary = manage_memory(history, long_term_summary)

    search_query = rewrite_query(query, history)
    retrieval = retrieve(search_query, notes, collection, graph, top_k=6)

    for _ in range(MAX_RETRIEVAL_ROUNDS - 1):
        if is_context_sufficient(query, retrieval['context']):
            break
        print("[retrieving more context...]")
        subqueries = generate_subqueries(query, retrieval['context'])
        existing_notes = {c['note_name'] for c in retrieval['context']}
        for sq in subqueries:
            extra = retrieve(sq, notes, collection, graph)
            for chunk in extra['context']:
                if chunk['note_name'] not in existing_notes:
                    retrieval['context'].append(chunk)
                    existing_notes.add(chunk['note_name'])
            retrieval['seed'] = list(set(retrieval['seed'] + extra['seed']))
            retrieval['expanded'] = list(set(retrieval['expanded'] + extra['expanded']))
    print()
    response = ask(query, retrieval, history, long_term_summary)

    # Source attribution block
    direct = retrieval['seed']
    related = retrieval['expanded']
    print("\n" + "─" * 40)
    print("Sources used:")
    if direct:
        print(f"  Direct match:  {', '.join(direct)}")
    if related:
        print(f"  Related notes: {', '.join(related)}")
    print("─" * 40 + "\n")

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})
