import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize persistent ChromaDB storage locally
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a fast lightweight embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)

def flatten_json_object(item: dict) -> str:
    """Converts a single JSON record into a readable prose block."""
    lines = []
    for key, value in item.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)

def process_large_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle top-level list or nested array
    records = data if isinstance(data, list) else data.get("items", [])
    
    documents = []
    metadatas = []
    ids = []

    for index, record in enumerate(records):
        doc_text = flatten_json_object(record)
        documents.append(doc_text)
        metadatas.append({"source_id": str(record.get("id", index))})
        ids.append(f"doc_{index}")

    # Upsert in batch
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully indexed {len(documents)} items into ChromaDB.")

if __name__ == "__main__":
    process_large_json("business_loans_knowledge_base_cleaned.json")