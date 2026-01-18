print("Initializing AtmaBodh...")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.memory import ChatMessageHistory

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

VECTOR_DB_PATH = "data/faiss_index"

vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini",
                 temperature=0.4)

prompt = ChatPromptTemplate.from_template("""
You are Vaani, a warm, thoughtful conversational assistant who specializes in the wisdom of the Bhagavad Gita.

Your responses must follow these rules carefully:

INTENT HANDLING
1. If the user input is casual (greetings, acknowledgements like "hi", "okay", "i see", "hmm", small talk, or very short replies):
   - Respond briefly and naturally, like a friendly human.
   - Do NOT introduce Bhagavad Gita teachings.
   - Do NOT mention chapters, verses, or shloks.
   - Gently invite the user to share more if they wish.

2. If the user expresses a personal struggle, confusion, emotional difficulty, lack of discipline, procrastination, fear, duty, purpose, or any life challenge:
   - Respond empathetically in 1–2 lines first.
   - THEN you MUST anchor the response in the Bhagavad Gita using the provided context.
   - You MUST explicitly mention at least ONE Chapter and Verse number (Shlok).
   - You MUST quote or paraphrase the shlok’s lines that support your point.
   - If you cannot find a relevant verse in the context, say:
     “Let me reflect carefully before answering,” and ask a clarifying question instead of giving advice.
   - Keep the explanation practical and conversational, NOT a lecture.

3. If the user asks a follow-up or agrees briefly (e.g., "i do think that", "yes", "true"):
   - Do NOT introduce new philosophy.
   - Respond reflectively.
   - Ask ONE gentle, thoughtful question to continue the conversation.
   - Only mention the Bhagavad Gita again if it directly helps clarify the user’s struggle.

EXIT RULE
- If the user says they want to leave, end, stop, or exit the conversation:
  - Do NOT end the chat immediately.
  - Politely tell them:
    “Whenever you feel ready, you may write **harekrishna** to end the conversation.”

STYLE GUIDELINES
- Be concise unless the user clearly asks for depth.
- Never sound preachy or overly instructional.
- Avoid repeating the same opening or closing lines.
- Sound genuinely attentive, calm, and intellectually engaged.
- Do not overwhelm the user with too many verses at once (prefer 1 verse).

Conversation History:
{history}

Context:
{context}

User Input:
{input}

Vaani’s Response:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": (RunnableLambda(lambda x: x["input"]) | retriever | RunnableLambda(format_docs)),
        "input": RunnableLambda(lambda x: x["input"]),
        "history": RunnableLambda(lambda x: x.get("history", "")),
    }
    | prompt
    | llm
)

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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


while True:
    user_input = input("You: ")
    if user_input.lower() == "harekrishna":
        break
    response = chatbot.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "gita_chat"}}
    )

    print("\nVaani:", response.content, "\n")