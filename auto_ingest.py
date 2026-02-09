import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
VECTOR_STORE = "vector_store"
INDEX_FILE = os.path.join(VECTOR_STORE, "index.faiss")
TEXT_FILE = os.path.join(VECTOR_STORE, "texts.pkl")

def split_into_chunks(text, chunk_size=2):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def auto_ingest():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = split_into_chunks(text)
                all_chunks.extend(chunks)

    embeddings = model.encode(all_chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(VECTOR_STORE, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(TEXT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print("Auto ingestion completed with chunking.")
    print(f"Total chunks indexed: {len(all_chunks)}")

if __name__ == "__main__":
    auto_ingest()
