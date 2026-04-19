# from google import genai
# from google.genai import types
import os
from ollama import Client
import dotenv

dotenv.load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_API = os.getenv("OLLAMA_API")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

SHORT_TERM_WINDOW = 6   # number of recent messages to keep verbatim
COMPRESS_THRESHOLD = 10  # compress when history exceeds this many messages


def _make_client() -> Client:
    return Client(host=OLLAMA_HOST, headers={'Authorization': f"Bearer {OLLAMA_API}"})

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


def compress_history(old_messages: list, existing_summary: str) -> str:
    """Summarize old messages into the long-term memory string."""
    client = _make_client()
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
    prompt = (
        f"Existing summary:\n{existing_summary}\n\n" if existing_summary else ""
    ) + (
        f"New conversation to add:\n{history_text}\n\n"
        "Update the summary to capture only the important facts and topics discussed. Be concise.\n"
        "Summary:"
    )
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response.response.strip()


def manage_memory(history: list, long_term_summary: str) -> tuple[list, str]:
    """Compress old messages into long-term summary when history grows too large."""
    if len(history) <= COMPRESS_THRESHOLD:
        return history, long_term_summary

    to_compress = history[:-SHORT_TERM_WINDOW]
    short_term = history[-SHORT_TERM_WINDOW:]
    long_term_summary = compress_history(to_compress, long_term_summary)
    return short_term, long_term_summary


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

    client = _make_client()

    prompt = (
        f"Conversation:\n{history_text}\n"
        f"Follow-up: {query}\n\n"
        "Rewrite the follow-up as a standalone search query using only the topic from the conversation. No explanation.\n"
        "Query:"
    )

    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    rewritten = response.response.strip() or query
    return rewritten

    # --- Gemini version (commented out) ---
    # response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    # return response.text.strip()


def is_context_sufficient(query: str, context: list) -> bool:
    client = _make_client()
    context_text = "\n".join(f"{c['note_name']}: {c['content']}" for c in context)
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "If the question asks about multiple items (e.g. 'explain each '), check whether EVERY "
        "specific item mentioned or implied by the question has a complete explanation in the context. "
        "If any item is missing or only partially covered, answer NO. "
        "Only answer YES if every aspect of the question can be completely answered from the context above. "
        "Reply YES or NO only."
    )
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response.response.strip().upper().startswith("YES")


def generate_subqueries(query: str, context: list) -> list[str]:
    client = _make_client()
    context_text = "\n".join(f"{c['note_name']}: {c['content']}" for c in context)
    prompt = (
        f"Question: {query}\n\n"
        f"Current context:\n{context_text}\n\n"
        "List the specific named topics, items, or concepts from the question that are NOT fully explained "
        "in the context above. For each missing item, write one short search query (the item name is enough). "
        "One query per line. No numbering, no explanation. Max 5 queries."
    )
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    lines = [l.strip() for l in response.response.strip().splitlines() if l.strip()]
    return lines[:3]


def ask(query: str, retrieval: dict, history: list = [], long_term_summary: str = "") -> str:
    system_prompt = build_system_prompt(retrieval)

    # Convert history from Ollama format to Gemini format
    # Ollama: {"role": "assistant", "content": "..."} → Gemini: {"role": "model", "parts": [{"text": "..."}]}
    # gemini_history = []
    # for msg in history:
    #     role = "model" if msg["role"] == "assistant" else msg["role"]
    #     gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

    if long_term_summary:
        system_prompt += f"\n\nConversation summary so far:\n{long_term_summary}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    client = _make_client()
    response = client.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        

    full_response = ""
    for chunk in response:
        token = chunk.message.content or ""
        print(token, end="", flush=True)
        full_response += token
        if chunk.done:
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
