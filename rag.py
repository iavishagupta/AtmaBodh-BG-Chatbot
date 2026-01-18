"""
rag.py — RAG module for AtmaBodh

- Loads FAISS vector store
- Loads OpenAI embeddings & ChatOpenAI LLM
- Loads prompt from external file
- Maintains per-session conversation memory
- Exposes `answer_question(question, session_id)` for FastAPI
"""

print("Initializing AtmaBodh...")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.memory import ChatMessageHistory

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/faiss_index")

vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini",
                 temperature=0.4)

PROMPT_PATH = os.getenv("PROMPT_PATH", "prompts/vaani_prompt.txt")

try:
    prompt_text = load_prompt(PROMPT_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Prompt file not found: {PROMPT_PATH}")

prompt = ChatPromptTemplate.from_template(prompt_text)

rag_chain = (
    {
        "context": (RunnableLambda(lambda x: x["input"]) | retriever | RunnableLambda(format_docs)),
        "input": RunnableLambda(lambda x: x["input"]),
        "history": RunnableLambda(lambda x: x.get("history", "")),
    }
    | prompt
    | llm
)

chatbot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("""🙏 ĀtmaBodh Conversation System :
        Hi, I am Vaani.❤️
        Feel free to ask any question and explore thoughtful and spiritual answers.
        Type 'HareKrishna' anytime to end the chat.\n""")


# ---------- PUBLIC API ----------
def answer_question(question: str, session_id: str = "gita_chat") -> str:
    """
    Main entry point for FastAPI or any other interface.
    """
    response = chatbot.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content
