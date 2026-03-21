from src.ingestion.vault_reader import load_notes
from src.retrieval.embedder import embed_vault, get_collection
import os
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve
from src.retrieval.graph_builder import build_graph
from src.agent.agent import ask, rewrite_query

load_dotenv()

notes = load_notes(os.getenv("VAULT_PATH"))
collection = get_collection()
embed_vault(notes, collection)
graph = build_graph(notes)

print("Obsidian RAG — ask anything about your notes (type 'exit' to quit)\n")

history = []
while True:
    query = input("> ")
    if query.strip() == "exit":
        break

    search_query = rewrite_query(query, history)
    retrieval = retrieve(search_query, notes, collection, graph)
    print()
    response = ask(query, retrieval, history)

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
