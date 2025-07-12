import json
import requests
import time

# --- Configuration ---
# Use our new, specialized model for creating embeddings
EMBEDDING_MODEL = "mxbai-embed-large" 
OLLAMA_API_URL = "http://127.0.0.1:11434/api/embeddings"
SOURCE_JSON_PATH = "knowledge_base.json"
OUTPUT_JSON_PATH = "vector_database.json"

def get_embedding(text, model_name):
    """Gets a vector embedding for a single piece of text from Ollama."""
    try:
        payload = {"model": model_name, "prompt": text}
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return None

def create_recursive_chunks(data_node, path_prefix=""):
    """
    Recursively walks through a JSON/dict structure and generates detailed,
    context-aware text chunks for embedding.
    """
    chunks = []
    # Base case: if the node is a simple value, create a chunk
    if isinstance(data_node, (str, int, float, bool)):
        if path_prefix:
            # Clean up the path for readability
            readable_path = path_prefix.replace('_', ' ').replace('.', ' -> ')
            content = f"Regarding '{readable_path}', the value is: {data_node}."
            chunks.append({"source": path_prefix, "content": content})
        return chunks

    # Recursive step for dictionaries
    if isinstance(data_node, dict):
        for key, value in data_node.items():
            # Construct the new path for the next level of recursion
            new_path = f"{path_prefix}.{key}" if path_prefix else key

            # Special handling for certain keys to create more natural sentences
            if key == 'name' and 'content' not in str(value): # Avoid re-chunking content
                parent_path_parts = path_prefix.split('.')
                context = parent_path_parts[-1].replace('_', ' ') if parent_path_parts else "entity"
                content = f"The name for the {context} is {value}."
                chunks.append({"source": new_path, "content": content})

            elif key == 'content':
                # If a key is 'content', treat its value as the primary chunk text
                chunks.append({"source": path_prefix, "content": str(value)})

            else:
                chunks.extend(create_recursive_chunks(value, new_path))

    # Recursive step for lists
    elif isinstance(data_node, list):
        for i, item in enumerate(data_node):
            # Create a chunk for each item in the list
            new_path = f"{path_prefix}[{i}]"

            # If the item in the list is a simple value (like a store name)
            if isinstance(item, str):
                singular_path = path_prefix.replace('_', ' ').rstrip('s') # e.g., 'store_locations' -> 'store location'
                content = f"A known {singular_path} is: {item}."
                chunks.append({"source": new_path, "content": content})
            else:
                # If the item is a dictionary or another list, recurse into it
                chunks.extend(create_recursive_chunks(item, new_path))

    return chunks

def main():
    """Main function to create the new, smarter vector database."""
    print("--- Starting Smart Vector Database Creation ---")

    try:
        with open(SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
        print(f"✅ Successfully loaded '{SOURCE_JSON_PATH}'.")
    except Exception as e:
        print(f"❌ Error loading source file: {e}")
        return

    print("🧠 Creating recursive chunks of data from the entire knowledge base...")
    # Call the new recursive function
    text_chunks = create_recursive_chunks(knowledge_data)
    print(f"✅ Created {len(text_chunks)} granular chunks.")

    # The rest of the main function remains the same...
    vector_database = []
    print(f"✨ Generating new embeddings using model '{EMBEDDING_MODEL}'...")

    for i, chunk in enumerate(text_chunks):
        content = chunk["content"]
        print(f"  - Processing chunk {i+1}/{len(text_chunks)}: {content[:70]}...")
        embedding = get_embedding(content, EMBEDDING_MODEL)

        if embedding:
            vector_database.append({
                "source": chunk["source"], 
                "content": content, 
                "vector": embedding,
                "created_at": 0.0
                })
            time.sleep(0.1)
        else:
            print(f"  - ⚠️ Failed to get embedding for chunk. Skipping.")

    print("\n💾 Saving new vector database to file...")
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(vector_database, f, indent=2)
        print(f"✅ Successfully saved new vector database to '{OUTPUT_JSON_PATH}'.")
    except Exception as e:
        print(f"❌ Error saving output file: {e}")

    print("\n--- Smart Vector Database Creation Complete! ---")
    print("IMPORTANT: Run `migrate_to_db.py` to upload the new vectors to MongoDB.")

if __name__ == "__main__":
    main()