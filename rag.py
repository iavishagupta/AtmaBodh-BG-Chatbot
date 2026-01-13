import os

chatbot = None

def initialize_rag():
    global chatbot
    if chatbot is not None:
        return chatbot

    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.document_loaders import DataFrameLoader
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_classic.memory import ChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_community.vectorstores import FAISS

    file_path = "bhagavad_gita_verses.csv"

    bg_data = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "madhurpant/bhagavad-gita-verses-dataset",
        file_path,
    )

    loader = DataFrameLoader(bg_data, page_content_column="translation")
    docs = loader.load()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    prompt = ChatPromptTemplate.from_template("""
    You are Vaani, an expert on the Bhagavad Gita.
    Use the conversation history and the context below.
    Answer ONLY using the context.
    Always mention Chapter and Verse (Shlok).
    Conversation History:
    {history}
    Context:
    {context}
    Question:
    {input}
    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["input"]) | retriever | RunnableLambda(format_docs),
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

    return chatbot


def answer_question(query: str, session_id: str = "bgita_chat"):
    bot = initialize_rag()
    response = bot.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content
