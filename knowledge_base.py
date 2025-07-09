import json
import numpy as np
import config
import llm_service

def load_vector_database():
    """Loads the vector database from the file specified in config."""
    try:
        with open(config.VECTOR_DATABASE_PATH, 'r', encoding='utf-8') as f:
            vector_database = json.load(f)
        print(f"✅ Successfully loaded vector database with {len(vector_database)} smart chunks.")
        return vector_database
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        return []

def save_vector_database(vector_database):
    """Saves the in-memory vector database back to the JSON file."""
    try:
        with open(config.VECTOR_DATABASE_PATH, 'w', encoding='utf-8') as f:
            json.dump(vector_database, f, indent=2)
        print("💾 Knowledge base successfully saved to disk.")
    except Exception as e:
        print(f"❌ Error saving knowledge base to file: {e}")

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def find_relevant_chunks(question_embedding, vector_database):
    """
    Finds the most relevant text chunks and their sources.
    Returns a tuple of (context_string, list_of_sources, top_score).
    """
    if not question_embedding:
        return "", [], 0.0
    
    similarities = []
    for i, chunk in enumerate(vector_database):
        score = cosine_similarity(question_embedding, chunk["vector"])
        similarities.append((score, i))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    top_score = similarities[0][0] if similarities else 0.0

    top_chunks = [vector_database[i] for score, i in similarities[:config.TOP_K] if score > 0]
    
    if not top_chunks:
        return "", [], 0.0

    context_str = "\n".join([chunk["content"] for chunk in top_chunks])
    sources = sorted(list(set([chunk["source"] for chunk in top_chunks])))
    
    return context_str, sources, top_score

def learn_new_fact(fact_text, vector_database):
    """
    Learns a new fact, adds it to the database, and saves it permanently.
    """
    print(f"🧠 Learning new fact: '{fact_text}'")
    
    new_embedding = llm_service.get_embedding(fact_text, config.EMBEDDING_MODEL)
    
    if new_embedding:
        vector_database.append({
            "source": "user_provided",
            "content": fact_text,
            "vector": new_embedding
        })
        print("✅ Understood! I've learned that new information.")
        save_vector_database(vector_database)
    else:
        print("❌ I'm sorry, I had trouble understanding that. Could you try rephrasing?")

def forget_fact(fact_text, vector_database):
    """
    Finds and removes a fact from the database based on semantic similarity,
    after user confirmation.
    """
    print(f"🔎 Searching for a fact similar to: '{fact_text}' to forget...")

    fact_embedding = llm_service.get_embedding(fact_text, config.EMBEDDING_MODEL)
    if not fact_embedding:
        print("❌ Could not create an embedding for the fact to forget. Please try rephrasing.")
        return

    similarities = []
    for i, chunk in enumerate(vector_database):
        score = cosine_similarity(fact_embedding, chunk["vector"])
        similarities.append((score, i, chunk["content"]))

    if not similarities:
        print("🤔 The knowledge base is empty. Nothing to forget.")
        return

    similarities.sort(key=lambda x: x[0], reverse=True)
    best_match_score, best_match_index, best_match_content = similarities[0]

    if best_match_score >= config.FORGET_SIMILARITY_THRESHOLD:
        print(f"🎯 Found a potential match with confidence {best_match_score:.2f}:")
        print(f"   '{best_match_content}'")
        
        try:
            confirmation = input("   Are you sure you want to permanently delete this? (yes/no): ").lower().strip()
            if confirmation == 'yes':
                vector_database.pop(best_match_index)
                print("✅ Fact forgotten successfully.")
                save_vector_database(vector_database)
            else:
                print("👍 Deletion cancelled.")
        except (KeyboardInterrupt, EOFError):
            print("\n👍 Deletion cancelled.")

    else:
        print(f"🤷 I couldn't find a confident match for that fact. The best match had a score of {best_match_score:.2f}.")
        print(f"   My closest guess was: '{best_match_content}'")
        print("   Nothing was deleted.")

def handle_correction(last_question, correction_text, vector_database):
    """
    Handles user corrections. It finds the chunk that was most likely responsible
    for the wrong answer to the last question and offers to replace it.
    """
    print("✍️ It looks like you're correcting me.")
    print(f"   New information provided: '{correction_text}'")

    if not last_question:
        print("\n   I don't have the context of the last question. I'll learn this as new information.")
        learn_new_fact(correction_text, vector_database)
        return

    # Find the chunk that was most relevant to the last question.
    question_embedding = llm_service.get_embedding(last_question, config.EMBEDDING_MODEL)
    if not question_embedding:
        print("\n   Could not process the last question. I'll learn this as new information.")
        learn_new_fact(correction_text, vector_database)
        return

    similarities = []
    for i, chunk in enumerate(vector_database):
        score = cosine_similarity(question_embedding, chunk["vector"])
        similarities.append((score, i, chunk["content"]))
    
    similarities.sort(key=lambda x: x[0], reverse=True)

    if not similarities:
        print("\n   The knowledge base is empty. Learning as new info.")
        learn_new_fact(correction_text, vector_database)
        return

    best_match_score, best_match_index, best_match_content = similarities[0]

    print(f"\n   I think this correction relates to the following fact in my knowledge:")
    print(f"   '{best_match_content}'")
    print(f"\n   Should I replace this fact with the new information you provided?")
    
    try:
        confirmation = input("   (yes/no): ").lower().strip()
        if confirmation == 'yes':
            new_embedding = llm_service.get_embedding(correction_text, config.EMBEDDING_MODEL)
            if not new_embedding:
                print("❌ I'm sorry, I had trouble understanding the new information. Nothing was changed.")
                return

            original_source = vector_database[best_match_index]['source']
            vector_database[best_match_index] = {
                "source": original_source,
                "content": correction_text,
                "vector": new_embedding
            }
            print("✅ Okay, I've updated my knowledge.")
            save_vector_database(vector_database)
        else:
            print("\n   Okay, I won't replace it. Would you like me to learn it as a completely new fact?")
            learn_as_new_confirmation = input("   (yes/no): ").lower().strip()
            if learn_as_new_confirmation == 'yes':
                learn_new_fact(correction_text, vector_database)
            else:
                print("👍 Okay, correction cancelled.")
    except (KeyboardInterrupt, EOFError):
        print("\n👍 Correction cancelled.") 