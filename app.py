from fastapi import FastAPI
from pydantic import BaseModel
import os
from rag import answer_question

app = FastAPI()

# ---------- Models ----------
class Query(BaseModel):
    question: str
    session_id: str = "gita_chat"

# ---------- Startup event ----------
@app.on_event("startup")
def startup_event():
    """
    Pre-warm FAISS and LLM into memory on server start.
    This avoids long response time for the first user.
    """
    print("Prewarming RAG system...")
    if os.getenv("PREWARM", "true").lower() == "true":
        try:
            answer_question("Hello!", session_id="warmup")
            print("RAG system pre-warmed successfully!")
        except Exception as e:
            print(f"Error during pre-warm: {e}")

# ---------- API Endpoints ----------
@app.post("/ask")
def ask(q: Query):
    """
    Main endpoint to ask questions.
    Expects JSON:
    {
        "question": "Your question here",
        "session_id": "optional_session_id"
    }
    """
    try:
        answer = answer_question(q.question, q.session_id)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "AtmaBodh BG chatbot API running"}
