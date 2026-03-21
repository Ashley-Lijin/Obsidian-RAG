import requests
import json
# from google import genai
# from google.genai import types
import os
import dotenv

dotenv.load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# GEMINI_API = os.getenv("GEMINI_API")
# GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# client = genai.Client(api_key=GEMINI_API)


def build_system_prompt(retrieval: dict) -> str:
    sources_text = ""
    for note in retrieval['context']:
        sources_text += f"=== {note['note_name']} ===\n{note['content']}\n\n"

    return f"""You are a research assistant. Answer using ONLY the notes below.

Rules:
- Use ONLY information from the notes. Do not add anything from outside.
- Examples must come directly from the notes. Do NOT invent examples.
- Do not repeat the same point twice.
- Fix grammar when presenting the content, but keep the meaning identical.
- If the answer is not in the notes, say "I don't have that in my notes."

Notes:
{sources_text}"""


def rewrite_query(query: str, history: list) -> str:
    """Rewrite a follow-up query into a self-contained search query using recent history."""
    if not history:
        return query

    # Build a short context window — last 4 messages (2 turns)
    recent = history[-4:]
    history_text = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = (
        f"Conversation:\n{history_text}\n"
        f"Follow-up: {query}\n\n"
        "Rewrite the follow-up as a standalone search query using only the topic from the conversation. No explanation.\n"
        "Query:"
    )

    response = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    })
    rewritten = response.json().get("response", query).strip()
    return rewritten

    # --- Gemini version (commented out) ---
    # response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    # return response.text.strip()


def ask(query: str, retrieval: dict, history: list = []) -> str:
    system_prompt = build_system_prompt(retrieval)

    # Convert history from Ollama format to Gemini format
    # Ollama: {"role": "assistant", "content": "..."} → Gemini: {"role": "model", "parts": [{"text": "..."}]}
    # gemini_history = []
    # for msg in history:
    #     role = "model" if msg["role"] == "assistant" else msg["role"]
    #     gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = requests.post(f"{OLLAMA_HOST}/api/chat", json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
    }, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            full_response += token
            if chunk.get("done"):
                break
    print()

    return full_response

    # --- Gemini version (commented out) ---
    # chat = client.chats.create(
    #     model=GEMINI_MODEL,
    #     config=types.GenerateContentConfig(system_instruction=system_prompt),
    #     history=gemini_history,
    # )
    # response = chat.send_message_stream(query)
    # full_response = ""
    # for chunk in response:
    #     token = chunk.text
    #     print(token, end="", flush=True)
    #     full_response += token
    # print()
    # return full_response
