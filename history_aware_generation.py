from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# Rewrites the user's question into a standalone question using chat history,
# so retrieval works correctly even for follow-up questions like "tell me more".
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Given the chat history and the latest user question, "
        "reformulate the question as a standalone question that can be understood "
        "without the chat history. Return ONLY the reformulated question."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Answers the question using retrieved context and the full conversation history.
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Use the retrieved context to answer the question. "
        "If you don't know the answer, say so — do not make one up.\n\n"
        "Context:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def load_vectorstore(persist_path="chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=persist_path, embedding_function=embeddings)

def build_chain(vectorstore, model="llama3.2", k=3):
    llm = ChatOllama(model=model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def contextualize_and_retrieve(input_dict):
        # On the first turn there's no history, so use the question as-is.
        # On follow-up turns, rewrite the question before retrieving.
        if input_dict.get("chat_history"):
            question = (CONTEXTUALIZE_PROMPT | llm | StrOutputParser()).invoke(input_dict)
        else:
            question = input_dict["input"]
        return retriever.invoke(question)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain: retrieve relevant chunks → inject as context → answer with history
    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(contextualize_and_retrieve) | RunnableLambda(format_docs)
        )
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

def chat(chain):
    # Accumulates HumanMessage / AIMessage pairs passed into every chain invocation.
    chat_history = []
    print("Chat started. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer = chain.invoke({"input": question, "chat_history": chat_history})
        print(f"\nAssistant: {answer}\n")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

def main():
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)
    chat(chain)

if __name__ == "__main__":
    main()
