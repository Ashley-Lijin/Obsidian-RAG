import subprocess

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


def ask(query: str, retrieval: dict) -> str:
    prompt = build_prompt(query, retrieval)
    result = subprocess.run(['gemini', '-p', prompt], capture_output=True, text=True)
    return result.stdout