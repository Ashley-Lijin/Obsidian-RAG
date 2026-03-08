from pprint import pprint
from valult_reader import load_notes
from embedder import embed_vault, get_collection
import os
from dotenv import load_dotenv
from retriever import retrieve
from graph_builder import build_graph

load_dotenv()

notes = load_notes(os.getenv("VAULT_PATH"))
collection = get_collection()
embed_vault(notes, collection)
print(collection.count())

graph = build_graph(notes)

result = retrieve("what is a neural network", notes, collection, graph)
print("Seeds:", result['seed'])
print("Expanded:", result['expanded'])
print(f"Total context notes: {len(result['context'])}")

# print(list(graph.edges()))
# print(notes['Neural Network']['links'])
# print(notes['Tensor']['links'])