from mcp.server.fastmcp import FastMCP
from vault_reader import load_notes
from embedder import get_collection, embed_vault
from retriever import retrieve
from graph_builder import build_graph
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH")

notes = load_notes(VAULT_PATH)
collection = get_collection()
# if collection.count() == 0:
#     embed_vault(notes, collection)
graph = build_graph(notes)

mcp = FastMCP("obsidian-rag")

@mcp.tool()
def search_vault(query: str):
    """Search the Obsidian vault for notes relevant to the query"""
    retrieval = retrieve(query, notes, collection, graph)
    return retrieval

@mcp.tool()
def write_note(title: str, content: str):
    """Create a new markdown note in the Obsidian vault"""
    path = Path(VAULT_PATH) / f"{title}.md"
    with open(path, "w") as f:
        f.write(content)
    return f"Note '{title}' created successfully."

print("je")

if __name__ == "__main__":
    mcp.run()
