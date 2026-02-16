import os
import requests
from services.retriever import retrieve_context

# -------------------------
# ENVIRONMENT DETECTION
# -------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
IS_CLOUD = GROQ_API_KEY is not None

# -------------------------
# LOCAL OLLAMA SETTINGS
# -------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# -------------------------
# CLOUD GROQ SETTINGS
# -------------------------

if IS_CLOUD:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_MODEL = "llama-3.1-8b-instant"

def pdf_agent(question: str):

    context = retrieve_context(question)

    # If no context found, fallback to reasoning mode
    if not context.strip():
        context = "No direct document match found. Use domain knowledge and provide best possible guidance."

    prompt = f"""
You are an internal BusinessNext enterprise support assistant.

Use the provided context if relevant.
If context is weak, infer logically but do NOT hallucinate product features.
Provide procedural solution steps.
Maximum 6 concise bullet points.
Each bullet under 18 words.

Context:
{context}

Question:
{question}

Answer:
"""

    if IS_CLOUD:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise enterprise AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Groq API error: {str(e)}"

    else:
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 180
                    }
                },
                timeout=60
            )

            res.raise_for_status()
            return res.json()["response"]

        except Exception:
            return "LLM service unavailable. Please ensure Ollama is running locally."
