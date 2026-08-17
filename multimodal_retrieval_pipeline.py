from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_multimodal_db"
CLIP_MODEL = "clip-ViT-B-32"


class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name: str = CLIP_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


def search(query: str, k: int = 5):
    embeddings = CLIPEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = vectorstore.similarity_search(query, k=k)

    text_results = [r for r in results if r.metadata.get("type") == "text"]
    image_results = [r for r in results if r.metadata.get("type") == "image"]

    print(f"\nQuery: {query}\n")

    if text_results:
        print(f"--- Text results ({len(text_results)}) ---")
        for i, doc in enumerate(text_results):
            print(f"[{i+1}] {doc.metadata.get('source', 'unknown')}")
            print(f"     {doc.page_content[:300]}...")
            print()

    if image_results:
        print(f"--- Image results ({len(image_results)}) ---")
        for i, doc in enumerate(image_results):
            print(f"[{i+1}] {doc.metadata.get('image_path')} "
                  f"(source: {doc.metadata.get('source')}, page {doc.metadata.get('page')})")
        print()

    return text_results, image_results


def main():
    search("revenue chart")

if __name__ == "__main__":
    main()
