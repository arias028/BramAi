import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# --- Configuration ---
# Updated to use the database name you chose!
MONGO_URI = "mongodb://localhost:27017/" 
DATABASE_NAME = "BramAi"
COLLECTION_NAME = "knowledge" # A simple name for our collection
JSON_FILE_PATH = "knowledge_base.json"

def migrate_json_to_mongodb():
    """
    Connects to MongoDB, reads the JSON knowledge base, and inserts it
    into the 'BramAi' database.
    """
    print("--- Starting Database Migration ---")

    # Step 1: Connect to MongoDB
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ismaster') 
        print("✅ Successfully connected to MongoDB.")
    except ConnectionFailure as e:
        print(f"❌ Error: Could not connect to MongoDB.")
        print(f"Please make sure MongoDB is running.")
        print(f"Details: {e}")
        return

    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Step 2: Clear any old data to ensure a fresh start
    if COLLECTION_NAME in db.list_collection_names():
        print(f"Collection '{COLLECTION_NAME}' already exists. Clearing old data...")
        collection.delete_many({})
    else:
        print(f"Creating new collection: '{COLLECTION_NAME}'")

    # Step 3: Read the JSON file
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
            print(f"✅ Successfully loaded '{JSON_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{JSON_FILE_PATH}' was not found in D:\Ai\BramAi.")
        print("Please make sure it's in the same folder as this script.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{JSON_FILE_PATH}' is not a valid JSON file.")
        return

    # Step 4: Insert the data into the collection
    try:
        # We insert the single large JSON object as one document in the collection.
        # This makes it easy to retrieve the whole knowledge base later.
        collection.insert_one(knowledge_data)
        print(f"✅ Successfully inserted knowledge into '{DATABASE_NAME}.{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"❌ An error occurred during data insertion: {e}")
        return

    print("\n--- Migration Complete! ---")
    client.close()

if __name__ == "__main__":
    migrate_json_to_mongodb()