from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "gita_chat"

@app.post("/ask")
def ask(q: Query):
    return {
        "answer": answer_question(q.question, q.session_id)
    }

@app.get("/")
def root():
    return {"status": "Bhagavad Gita chatbot API running"}

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

