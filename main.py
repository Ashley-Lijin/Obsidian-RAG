from valult_reader import load_notes
from embedder import embed_vault, get_collection
import os
from dotenv import load_dotenv

load_dotenv()

notes = load_notes(os.getenv("VAULT_PATH"))
collection = get_collection()
embed_vault(notes, collection)
print(collection.count())

results = collection.query(
    query_texts=["what is machine learning"],
    n_results=3
)

for i, doc in enumerate(results['documents'][0]):
    print(f"--- Result {i+1} ---")
    print(results['metadatas'][0][i]['note_name'])
    print(doc[:200])
    print()