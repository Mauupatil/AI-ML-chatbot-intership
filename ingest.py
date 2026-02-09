import faiss
import pickle
from sentence_transformers import SentenceTransformer

def ingest_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("data/medical.txt", "r") as f:
        texts = f.readlines()

    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "vector_store/index.faiss")

    with open("vector_store/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

if __name__ == "__main__":
    ingest_data()