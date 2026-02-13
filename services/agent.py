from services.retriever import retrieve_context
import requests
from core.config import OLLAMA_MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"

SIMILARITY_THRESHOLD = 0.75  # deterministic fast-path threshold


def format_steps_directly(context: str):
    """
    Deterministic fast path.
    Extract lines that look like steps.
    """
    lines = context.split("\n")
    steps = []

    for line in lines:
        line = line.strip()
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
            steps.append(line)
        elif line.lower().startswith(("click", "navigate", "select", "log")):
            steps.append(line)

        if len(steps) >= 6:
            break

    if not steps:
        return None

    return "\n".join(f"- {s}" for s in steps)


def pdf_agent(question: str):

    context, scores = retrieve_context(question, return_scores=True)

    if not context:
        return "Not found in documents."

    max_score = max(scores) if scores else 0

    # ------------------------------------------------
    # 🚀 FAST DETERMINISTIC PATH
    # ------------------------------------------------
    if max_score >= SIMILARITY_THRESHOLD:
        formatted = format_steps_directly(context)
        if formatted:
            return formatted

    # ------------------------------------------------
    # 🧠 LIGHT LLM FALLBACK
    # ------------------------------------------------
    prompt = f"""
You are an internal BusinessNext support assistant.

Answer in maximum 6 short bullet points.
Each bullet under 15 words.
Be concise and procedural.
If not found, say: Not found in documents.

Context:
{context}

Question:
{question}

Answer:
"""

    res = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 120
            }
        },
        timeout=60
    )

    res.raise_for_status()
    return res.json()["response"]
