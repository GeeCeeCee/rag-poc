import os
from itertools import groupby
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="documents"):
    """Load all the text file from the directory"""
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No pdf files are present in the given path - {docs_path}")

    # for i, doc in enumerate(documents):
    #     print(f"Index - {i}")
    #     print(f"Source - {doc.metadata['source']}")
    #     print(f"Content Length - {len(doc.page_content)} characters")
    #     print(f"Metadata - {doc.metadata}")


    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    # Merge all pages per PDF into one document so splits follow paragraph
    # breaks rather than page boundaries
    merged = []
    for source, pages in groupby(documents, key=lambda d: d.metadata.get("source")):
        pages = list(pages)
        merged.append(Document(
            page_content="\n\n".join(p.page_content for p in pages),
            metadata={"source": source}
        ))

    splitter = SemanticChunker(
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        breakpoint_threshold_amount=70,
        breakpoint_threshold_type="percentile"
    )
    chunks = splitter.split_documents(merged)
    print(f"Merged {len(documents)} pages into {len(merged)} documents, split into {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks, persist_path="chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{persist_path}'")
    return vectorstore


def main():
    print("Main Function")
    documents = load_documents("documents")
    chunks = split_documents(documents)
    create_vectorstore(chunks)

if __name__ == "__main__":
    main()