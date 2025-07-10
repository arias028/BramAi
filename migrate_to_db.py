import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import config # Import our central config

def migrate_json_to_mongodb():
    """
    Connects to MongoDB using settings from config.py, reads the
    vector_database.json file, and inserts each item as a separate
    document into the specified collection.
    """
    print("--- Starting Database Migration ---")

    # Step 1: Connect to MongoDB
    try:
        print(f"Connecting to MongoDB at {config.MONGO_URI}...")
        client = MongoClient(config.MONGO_URI)
        client.admin.command('ismaster')
        print("✅ Successfully connected to MongoDB.")
    except ConnectionFailure as e:
        print(f"❌ Error: Could not connect to MongoDB.")
        print("Please ensure your MONGO_URI in 'config.py' is correct.")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during connection: {e}")
        return

    db = client[config.DATABASE_NAME]
    collection = db[config.COLLECTION_NAME]

    # Step 2: Clear any old data to ensure a fresh start
    print(f"Checking for existing data in '{config.COLLECTION_NAME}'...")
    count = collection.count_documents({})
    if count > 0:
        print(f"Found {count} existing documents. Clearing collection...")
        collection.delete_many({})
    else:
        print(f"Collection is empty. Ready for new data.")

    # Step 3: Read the JSON file
    # We now use the original vector_database.json used by the main app
    json_file_path = "vector_database.json" 
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
            # Ensure the data is a list
            if not isinstance(knowledge_data, list):
                print(f"❌ Error: Expected '{json_file_path}' to contain a JSON list of objects.")
                return
            print(f"✅ Successfully loaded {len(knowledge_data)} documents from '{json_file_path}'.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    # Step 4: Insert the data into the collection
    if not knowledge_data:
        print("⚠️ The knowledge base file is empty. Nothing to migrate.")
        return
        
    try:
        print(f"Inserting {len(knowledge_data)} documents into '{config.DATABASE_NAME}.{config.COLLECTION_NAME}'...")
        collection.insert_many(knowledge_data)
        print(f"✅ Successfully inserted all documents.")
    except Exception as e:
        print(f"❌ An error occurred during data insertion: {e}")
        return

    print("\n--- Migration Complete! ---")
    print("\nNext Steps:")
    print("1. IMPORTANT: Create a vector search index in your MongoDB Atlas collection.")
    print("2. Run bram_ai.py to start the application. It will guide you if the index is missing.")
    
    client.close()

if __name__ == "__main__":
    migrate_json_to_mongodb()