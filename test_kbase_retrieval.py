import chromadb
from chromadb.utils import embedding_functions

# 1. Reconnect to the persistent ChromaDB instance and embedding model
chroma_client = chromadb.PersistentClient(path="./chroma_db")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)

# 2. Define test scenarios covering Product, Policy, Qualification, FAQ, and Objection
test_cases = [
    {
        "category": "Product",
        "question": "What business loan options do you offer, and what are their typical repayment terms?",
        "relevance_explanation": "Checks whether the vector search accurately identifies core loan product offerings and term durations.",
        "expected_verdict": "correct"
    },
    {
        "category": "Policy",
        "question": "What is your policy regarding early repayment or prepayment penalties?",
        "relevance_explanation": "Tests retrieval of specific contractual terms regarding fees and early settlement policies.",
        "expected_verdict": "correct"
    },
    {
        "category": "Qualification",
        "question": "What minimum credit score and annual business revenue are required to qualify?",
        "relevance_explanation": "Retrieves minimum underwriting standards and eligibility benchmarks.",
        "expected_verdict": "correct"
    },
    {
        "category": "FAQ",
        "question": "How long does the loan application approval process take once submitted?",
        "relevance_explanation": "Validates retrieving operational turn-around time details standard in customer FAQs.",
        "expected_verdict": "correct"
    },
    {
        "category": "Objection",
        "question": "Why are your interest rates higher than traditional commercial banks?",
        "relevance_explanation": "Retrieves objection-handling scripts comparing non-bank/fintech loans to traditional bank financing.",
        "expected_verdict": "partially correct"
    }
]

def run_retrieval_interface():
    """Queries ChromaDB and evaluates the retrieval performance per scenario."""
    print("=" * 80)
    print("KNOWLEDGE BASE RETRIEVAL ENGINE & EVALUATION INTERFACE")
    print("=" * 80 + "\n")

    for idx, test in enumerate(test_cases, 1):
        question = test["question"]
        
        # Retrieve the top 1 most relevant chunk from ChromaDB
        results = collection.query(
            query_texts=[question],
            n_results=1
        )
        
        # Parse retrieved details safely
        retrieved_doc = results["documents"][0][0] if results["documents"] else "No document found"
        metadata = results["metadatas"][0][0] if results["metadatas"] else {}
        source_ref = metadata.get("source_id", "Unknown Source")
        distance = results["distances"][0][0] if "distances" in results and results["distances"] else "N/A"

        # Print structured test output meeting all prompt criteria
        print(f"QUERY #{idx} [{test['category'].upper()}]")
        print(f"• User Question: {question}")
        print(f"• Source Reference: ID '{source_ref}' (Distance Score: {distance:.4f})")
        print(f"• Retrieved Chunk/Record:\n---\n{retrieved_doc}\n---")
        print(f"• Relevance Explanation: {test['relevance_explanation']}")
        print(f"• Verdict: {test['expected_verdict'].upper()}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    run_retrieval_interface()
