from valult_reader import load_notes
from embedder import embed_vault, get_collection
import os
from dotenv import load_dotenv
from retriever import retrieve
from graph_builder import build_graph
from agent import ask

load_dotenv()

notes = load_notes(os.getenv("VAULT_PATH"))
collection = get_collection()
embed_vault(notes, collection)
print(collection.count())

graph = build_graph(notes)

query = "say an example where i spoted confirmation bias recently"
retrieval = retrieve(query, notes, collection, graph)
response = ask(query, retrieval)
print(response)