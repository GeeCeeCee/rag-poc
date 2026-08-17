import os
import base64
from io import BytesIO
from itertools import groupby

import fitz  # pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR = "images"
CHROMA_PATH = "chroma_multimodal_db"

# CLIP encodes text and images in the same vector space, enabling cross-modal retrieval.
CLIP_MODEL = "clip-ViT-B-32"


class CLIPEmbeddings(Embeddings):
    """LangChain-compatible wrapper around CLIP's text encoder."""

    def __init__(self, model_name: str = CLIP_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_image(self, image: Image.Image) -> list[float]:
        return self.model.encode(image).tolist()


def load_documents(docs_path="documents"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory not found: {docs_path}")

    loader = DirectoryLoader(path=docs_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No PDFs found in {docs_path}")

    return documents


def extract_and_embed_text(documents, embeddings: CLIPEmbeddings, chunk_size=300, chunk_overlap=30):
    """Merge pages per PDF, split into chunks, return LangChain Documents."""
    merged = []
    for source, pages in groupby(documents, key=lambda d: d.metadata.get("source")):
        pages = list(pages)
        merged.append(Document(
            page_content="\n\n".join(p.page_content for p in pages),
            metadata={"source": source, "type": "text"}
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(merged)
    print(f"Text: {len(merged)} PDFs → {len(chunks)} chunks")
    return chunks


def extract_images_from_pdfs(docs_path="documents") -> list[dict]:
    """Extract embedded images from all PDFs using pymupdf."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    image_records = []

    for filename in os.listdir(docs_path):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(docs_path, filename)
        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf):
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]

                image_filename = f"{filename}_p{page_num}_img{img_index}.png"
                image_path = os.path.join(IMAGES_DIR, image_filename)

                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image.save(image_path)

                image_records.append({
                    "image": image,
                    "image_path": image_path,
                    "source": pdf_path,
                    "page": page_num,
                })

        pdf.close()

    print(f"Images: extracted {len(image_records)} images")
    return image_records


def create_vectorstore(text_chunks, image_records, embeddings: CLIPEmbeddings):
    # Store text chunks via LangChain's Chroma wrapper
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Add image embeddings directly via the underlying ChromaDB collection
    # (LangChain's wrapper only handles text; images need direct access)
    if image_records:
        collection = vectorstore._collection
        ids = [f"img_{i}" for i in range(len(image_records))]
        image_embeddings = [embeddings.embed_image(r["image"]) for r in image_records]
        metadatas = [
            {"type": "image", "source": r["source"], "page": r["page"], "image_path": r["image_path"]}
            for r in image_records
        ]
        # ChromaDB requires a documents field; use a placeholder
        documents = [f"[IMAGE: {r['image_path']}]" for r in image_records]

        collection.add(
            ids=ids,
            embeddings=image_embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"Images: stored {len(image_records)} image embeddings")

    total = (vectorstore._collection.count())
    print(f"Total records in ChromaDB: {total}")
    return vectorstore


def main():
    embeddings = CLIPEmbeddings()

    documents = load_documents("documents")
    text_chunks = extract_and_embed_text(documents, embeddings)
    image_records = extract_images_from_pdfs("documents")
    create_vectorstore(text_chunks, image_records, embeddings)


if __name__ == "__main__":
    main()
