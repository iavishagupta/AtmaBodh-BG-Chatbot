from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "gita_chat"

@app.post("/ask")
def ask(q: Query):
    return {"answer": answer_question(q.question, q.session_id)}

@app.get("/")
def root():
    return {"status": "AtmaBodh BG chatbot API running"}
