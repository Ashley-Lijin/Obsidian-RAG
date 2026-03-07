from pathlib import Path
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def load_notes(vault_path: str) -> dict[str, dict[str, str | list[str]]]:
    vault = {}
    file = Path(vault_path)
    for note in file.rglob("*.md"):
        key = note.stem
        content = note.read_text(encoding="utf-8")
        vault[key] = {
            "content": content,
            "links": extract_links(content)
        }
    return vault



def extract_links(context:str) -> list[str]:
    pattern = r'\[\[([^|\]]+)(?:\|[^\]]*)?\]\]'
    links = re.findall(pattern, context)
    return links


notes = load_notes(os.getenv("VAULT_PATH"))