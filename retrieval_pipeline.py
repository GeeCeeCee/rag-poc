from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """Use the following context to answer the question.
If you don't know the answer, say you don't know — do not make one up.

Context:
{context}

Question: {question}

Answer:"""

def load_vectorstore(persist_path="chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=persist_path, embedding_function=embeddings)

def build_chain(vectorstore, model="llama3.2", k=2):
    llm = OllamaLLM(model=model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def search(vectorstore, question, k=2):
    results = vectorstore.similarity_search(question, k=k)
    print(f"\nQuestion: {question}\n")
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})")
        print(f"     {doc.page_content[:300]}...")
        print()
    return results

def ask(chain, question):
    print(f"\nQuestion: {question}\n")
    answer = chain.invoke(question)
    print(f"Answer: {answer}\n")
    return answer

def main():
    vectorstore = load_vectorstore()
    search(vectorstore, "How much did Microsoft pay to acquire github?")

    chain = build_chain(vectorstore)
    ask(chain, "What is the annual revenue of Microsoft over the years?")

if __name__ == "__main__":
    main()
