import chromadb
from chromadb.utils import embedding_functions
from src.ingestion.chunker import chunk_text
import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path("/Users/ashleylijin/Developer/Obsidian RAG/.chroma_db/manifest.json")


def get_collection(collection_name: str = "Obsidain_valult"):
    chroma_client = chromadb.PersistentClient(path='/Users/ashleylijin/Developer/Obsidian RAG/.chroma_db')
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

    return collection


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def embed_vault(notes: dict, collection):
    manifest = _load_manifest()
    current_names = set(notes.keys())
    manifest_names = set(manifest.keys())

    # Delete chunks for notes removed from the vault
    removed = manifest_names - current_names
    for name in removed:
        existing = collection.get(where={"note_name": name})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        del manifest[name]

    # Embed only new or changed notes
    for note_name, note_data in notes.items():
        content = note_data['content']
        h = _hash(content)
        if manifest.get(note_name) == h:
            continue  # unchanged — skip

        chunks = chunk_text(content)
        safe_name = note_name.replace(" ", "_")

        ids = [f"{safe_name}__chunk_{i}" for i in range(len(chunks))]
        documents = chunks
        metadatas = [
            {"note_name": note_name, "chunk_index": i, "links": ",".join(note_data['links'])}
            for i in range(len(chunks))
        ]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        manifest[note_name] = h

    _save_manifest(manifest)
