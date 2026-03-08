
def chunk_text(content: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    chunks = []
    words = content.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
