import faiss
import pickle
from sentence_transformers import SentenceTransformer

def retrieve(query, top_k=2):
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load FAISS index
    index = faiss.read_index("vector_store/index.faiss")

    # Load original texts
    with open("vector_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    # Convert query to embedding
    query_embedding = model.encode([query])

    # Search in vector DB
    distances, indices = index.search(query_embedding, top_k)

    # Return matched texts
    results = [texts[i] for i in indices[0]]
    return results


# Test block
if __name__ == "__main__":
    question = "What is Artificial Intelligence?"
    results = retrieve(question)
    print("Question:", question)
    print("Retrieved Context:")
    for r in results:
        print("-", r)
        