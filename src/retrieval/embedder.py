from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
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
_sparse_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_embedder


def _str_to_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def get_collection(collection_name: str = COLLECTION_NAME):
    client = QdrantClient(path=DB_PATH)
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        info = client.get_collection(collection_name)
        has_sparse = info.config.params.sparse_vectors is not None
        if not has_sparse:
            # Old single-vector schema — recreate for hybrid support
            client.delete_collection(collection_name)
            existing.discard(collection_name)
            if MANIFEST_PATH.exists():
                MANIFEST_PATH.write_text("{}")

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
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
    sparse_embedder = get_sparse_embedder()
    manifest = _load_manifest()
    current_names = set(notes.keys())
    manifest_names = set(manifest.keys())

    for name in manifest_names - current_names:
        _delete_note_chunks(client, collection_name, name)
        del manifest[name]

    for note_name, note_data in notes.items():
        content = note_data["content"]
        h = _hash(content)
        if manifest.get(note_name) == h:
            continue

        chunks = chunk_text(content)
        safe_name = note_name.replace(" ", "_")

        _delete_note_chunks(client, collection_name, note_name)

        dense_embeddings = embedder.encode(chunks).tolist()
        sparse_embeddings = list(sparse_embedder.embed(chunks))

        points = []
        for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
            chunk_id = f"{safe_name}__chunk_{i}"
            points.append(
                models.PointStruct(
                    id=_str_to_uuid(chunk_id),
                    vector={
                        "dense": dense,
                        "sparse": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "note_name": note_name,
                        "chunk_index": i,
                        "links": ",".join(note_data["links"]),
                        "text": chunk,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        manifest[note_name] = h

    _save_manifest(manifest)
