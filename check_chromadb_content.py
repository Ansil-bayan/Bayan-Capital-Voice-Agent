import chromadb

# 1. Connect to your persistent directory
client = chromadb.PersistentClient(path="./chroma_db")  # Replace with your path

# 2. List all collections to find your collection name
print(client.list_collections())

# 3. Get your specific collection
collection = client.get_collection(name="knowledge_base")  # Replace with your collection name

# 4. Peek at the first few records (default is 10)
print(collection.peek())

# Or retrieve all documents, IDs, and metadata
all_data = collection.get()
print(all_data)