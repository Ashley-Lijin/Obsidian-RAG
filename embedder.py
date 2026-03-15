import chromadb
from chromadb.utils import embedding_functions
from chunker import chunk_text

def get_collection(collection_name: str = "Obsidain_valult"):
    chroma_client = chromadb.PersistentClient(path='/Users/ashleylijin/Developer/Obsidian RAG/.chroma_db')
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

    return collection

def embed_vault(notes: dict, collection):

    for note_name, note_data in notes.items():
        content = note_data['content']
        links = note_data['links']

        chunks = chunk_text(content)

        ids = []
        document = []
        metadata = []

        name = note_name.replace(" ", "_")


        for i, chunk in enumerate(chunks):
            ids.append(f"{name}__chunk_{i}")
            document.append(chunk)
            metadata.append({
                "note_name": note_name,
                "chunk_index": i,
                "links": ",".join(links)
            })


        collection.upsert(
            ids=ids,
            documents=document,
            metadatas=metadata
        )

