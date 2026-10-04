from langchain_ollama import ChatOllama

def get_gemini_client():
    """Returns local Ollama client (kept name consistent so no other files need changes)."""
    return ChatOllama(
        model="llama3.2",
        temperature=0.0
    )