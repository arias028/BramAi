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
    Creates highly specific, sentence-like chunks to drastically improve
    semantic search accuracy and enable direct, factual answers.
    """
    chunks = []
    kb = data.get("knowledge_base", {})

    # Company Profile & Owner (CRITICAL)
    company_profile = kb.get("company_profile", {})
    if company_profile:
        # Find owner name from the management section
        management = kb.get("bina_team_profile", {}).get("management", {})
        for role, details in management.items():
            if "owner of laskar buah" in details.get("notes_en", "").lower():
                chunks.append({
                    "source": f"bina_team_profile.management.{role}.owner",
                    "content": f"Pemilik Laskar Buah adalah {details['name']}."
                })
                break

        bina_info = company_profile.get("entity_hierarchy", {})
        if bina_info.get("spinoff_year"):
            chunks.append({
                "source": "company_profile.entity_hierarchy.spinoff_year",
                "content": f"BINA (Bram Innovation Network and Access) resmi didirikan sebagai spinoff pada tahun {bina_info['spinoff_year']}."
            })
        # Add explicit relationship chunk
        chunks.append({
            "source": "company_profile.entity_hierarchy.relationship",
            "content": "BINA (Bram Innovation Network and Access) adalah perusahaan afiliasi dan merupakan spinoff dari Laskar Buah Group, sehingga keduanya adalah entitas yang berhubungan tetapi tidak sama."
        })

    # Team Profile
    team_profile = kb.get("bina_team_profile", {})
    management = team_profile.get("management", {})
    for role, details in management.items():
        role_title = role.replace('_', ' ').title()
        chunks.append({
            "source": f"bina_team_profile.management.{role}",
            "content": f"Posisi {role_title} di BINA dipegang oleh {details['name']}."
        })

    # Workflows
    workflows = kb.get("bina_operations", {})
    if workflows and workflows.get("hardware_request_flow"):
        chunks.append({
            "source": "bina_operations.hardware_request_flow",
            "content": f"Alur lengkap untuk permintaan hardware adalah: {workflows['hardware_request_flow']}"
        })

    # Store Locations (IMPORTANT)
    store_locations = kb.get("group_operational_structure", {}).get("store_locations", [])
    if store_locations:
        locations_str = ", ".join(store_locations[:15]) + " dan lain-lain"
        chunks.append({
            "source": "group_operational_structure.store_locations",
            "content": f"Beberapa lokasi toko Laskar Buah antara lain: {locations_str}."
        })
        
    print(f"Generated {len(chunks)} smart chunks.")
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
            vector_database.append({
                "source": chunk["source"], 
                "content": content, 
                "vector": embedding,
                "created_at": 0.0  # Add a default old timestamp for base knowledge
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

if __name__ == "__main__":
    main()