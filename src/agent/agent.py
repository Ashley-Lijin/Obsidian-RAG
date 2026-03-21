import requests
import os
import dotenv
import json

dotenv.load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

def build_prompt(query: str, retrieval: dict) -> str:
    context_text = ""
    for note in retrieval['context']:
        context_text += f"## {note['note_name']}\n{note['content']}\n\n"

    prompt = f"""You are an AI assistant with access to a personal knowledge base.

## Knowledge Base Context
{context_text}

## Query
{query}

## Instructions
- Answer using only the notes above
- Cite notes using [[Note Name]] syntax
- If the answer isn't in the notes, say 'IDK'
- If you need to use multiple notes, cite them all"""

    return prompt


def ask(query: str, retrieval: dict) -> None:
    prompt = build_prompt(query, retrieval)
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk.get("response", ""), end="", flush=True)
            if chunk.get("done"):
                break
    print()  # newline at the end
