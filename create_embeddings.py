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

def create_smart_chunks(data):
    """
    A smarter chunking function that tries to keep related information together.
    It creates a single text block for each logical entity.
    """
    chunks = []
    
    # Example of special handling for the management team
    management_team = data.get("knowledge_base", {}).get("bina_team_profile", {}).get("management", {})
    for role, details in management_team.items():
        # Create one coherent chunk for each manager
        chunk_text = f"Management Position: {role}, Name: {details.get('name')}"
        if details.get('notes_en'):
            chunk_text += f", Notes: {details.get('notes_en')}"
        if details.get('notes_id'):
            chunk_text += f", Catatan: {details.get('notes_id')}"
        if details.get('origin'):
            chunk_text += f", Origin: {details.get('origin')}"
        chunks.append({"source": f"bina_team_profile.management.{role}", "content": chunk_text})
        
    # Example for developers
    developers = data.get("knowledge_base", {}).get("bina_team_profile", {}).get("developers_and_specialists", [])
    for dev in developers:
        chunk_text = f"Developer: {dev.get('name')}, Role: {dev.get('role')}, Details: {dev.get('details')}"
        chunks.append({"source": f"bina_team_profile.developers.{dev.get('name')}", "content": chunk_text})

    # Add other key-value pairs as simple chunks
    def fallback_chunker(d, path=""):
        for key, value in d.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                # Avoid re-chunking sections we already handled
                if key not in ['management', 'developers_and_specialists']:
                    yield from fallback_chunker(value, new_path)
            elif isinstance(value, list):
                pass # You could add more list handling here if needed
            else:
                yield {"source": new_path, "content": f"{key}: {value}"}

    # Run the fallback chunker on the whole document to catch everything else
    all_chunks = set(c['content'] for c in chunks) # Use a set to avoid duplicates
    for chunk in fallback_chunker(data):
        if chunk['content'] not in all_chunks:
            chunks.append(chunk)
            all_chunks.add(chunk['content'])

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

    print("🧠 Creating smart chunks of data...")
    text_chunks = create_smart_chunks(knowledge_data)
    print(f"✅ Created {len(text_chunks)} smart chunks.")

    vector_database = []
    print(f"✨ Generating new embeddings using model '{EMBEDDING_MODEL}'...")
    
    for i, chunk in enumerate(text_chunks):
        content = chunk["content"]
        print(f"  - Processing chunk {i+1}/{len(text_chunks)}...")
        embedding = get_embedding(content, EMBEDDING_MODEL)
        
        if embedding:
            vector_database.append({"source": chunk["source"], "content": content, "vector": embedding})
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

if __name__ == "__main__":
    main()