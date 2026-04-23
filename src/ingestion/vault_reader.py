from pathlib import Path
import re
import json

NOTES_CACHE_PATH = Path("/Users/ashleylijin/Developer/Obsidian RAG/.qdrant_db/notes_cache.json")


def load_notes(vault_path: str) -> dict[str, dict[str, str | list[str]]]:
    vault_dir = Path(vault_path)

    # Load cache: {note_name: {content, links, mtime}}
    if NOTES_CACHE_PATH.exists():
        cache = json.loads(NOTES_CACHE_PATH.read_text())
    else:
        cache = {}

    vault = {}
    updated = False

    for note_path in vault_dir.rglob("*.md"):
        key = note_path.stem
        mtime = str(note_path.stat().st_mtime)

        if key in cache and cache[key]["mtime"] == mtime:
            # File unchanged — use cache
            vault[key] = {"content": cache[key]["content"], "links": cache[key]["links"]}
        else:
            # File new or changed — read from disk
            content = note_path.read_text(encoding="utf-8")
            links = extract_links(content)
            vault[key] = {"content": content, "links": links}
            cache[key] = {"content": content, "links": links, "mtime": mtime}
            updated = True

    # Remove deleted notes from cache
    deleted = set(cache.keys()) - set(vault.keys())
    for key in deleted:
        del cache[key]
        updated = True

    if updated:
        NOTES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTES_CACHE_PATH.write_text(json.dumps(cache, indent=2))

    return vault


def extract_links(context: str) -> list[str]:
    pattern = r'\[\[([^|\]]+)(?:\|[^\]]*)?\]\]'
    links = re.findall(pattern, context)
    return links
