from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from src.ingestion.chunker import chunk_text
import hashlib
import json
import uuid
from pathlib import Path

DB_PATH = "/Users/ashleylijin/Developer/Obsidian RAG/.qdrant_db"
MANIFEST_PATH = Path(DB_PATH) / "manifest.json"
COLLECTION_NAME = "obsidian_vault"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _str_to_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def get_collection(collection_name: str = COLLECTION_NAME):
    client = QdrantClient(path=DB_PATH)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    return client, collection_name


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def _delete_note_chunks(client: QdrantClient, collection_name: str, note_name: str):
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="note_name", match=models.MatchValue(value=note_name))]
            )
        ),
    )


def embed_vault(notes: dict, collection):
    client, collection_name = collection
    embedder = get_embedder()
    manifest = _load_manifest()
    current_names = set(notes.keys())
    manifest_names = set(manifest.keys())

    # Delete chunks for notes removed from the vault
    for name in manifest_names - current_names:
        _delete_note_chunks(client, collection_name, name)
        del manifest[name]

    # Embed only new or changed notes
    for note_name, note_data in notes.items():
        content = note_data["content"]
        h = _hash(content)
        if manifest.get(note_name) == h:
            continue

        chunks = chunk_text(content)
        safe_name = note_name.replace(" ", "_")

        # Remove stale chunks before upserting
        _delete_note_chunks(client, collection_name, note_name)

        embeddings = embedder.encode(chunks).tolist()
        points = [
            models.PointStruct(
                id=_str_to_uuid(f"{safe_name}__chunk_{i}"),
                vector=embedding,
                payload={
                    "note_name": note_name,
                    "chunk_index": i,
                    "links": ",".join(note_data["links"]),
                    "text": chunk,
                },
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        client.upsert(collection_name=collection_name, points=points)
        manifest[note_name] = h

    _save_manifest(manifest)
