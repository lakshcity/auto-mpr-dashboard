from services.retriever import retrieve_context
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

def pdf_agent(question: str):
    context = retrieve_context(question)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents".

Context:
{context}

Question:
{question}
"""

    res = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    res.raise_for_status()
    return res.json()["response"]
