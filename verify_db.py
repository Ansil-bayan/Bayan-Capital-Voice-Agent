import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

try:
    collection = chroma_client.get_collection(
        name="knowledge_base",
        embedding_function=sentence_transformer_ef
    )
    count = collection.count()
    print(f"SUCCESS: 'knowledge_base' collection found with {count} indexed records.")
    
    # Test a sample query
    sample_results = collection.query(query_texts=["interest rate loan eligibility"], n_results=2)
    print("Sample Retrieval Test:", sample_results["documents"])
    
except Exception as e:
    print("ERROR: Database collection empty or missing!", e)
    print("Action: Re-run 'python ingest.py' to populate ChromaDB.")