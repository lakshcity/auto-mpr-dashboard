import requests
import streamlit as st
from services.retriever import retrieve_context

# -------------------------
# ENVIRONMENT DETECTION
# -------------------------

IS_CLOUD = "GROQ_API_KEY" in st.secrets

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
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    GROQ_MODEL = "llama-3.1-8b-instant"


def pdf_agent(question: str):

    context = retrieve_context(question)

    if not context:
        return "Not found in documents."

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

    # -------------------------
    # CLOUD MODE (Groq)
    # -------------------------
    if IS_CLOUD:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful enterprise assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Groq API error: {str(e)}"

    # -------------------------
    # LOCAL MODE (Ollama)
    # -------------------------
    else:
        try:
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

        except Exception:
            return "LLM service unavailable. Please ensure Ollama is running locally."
