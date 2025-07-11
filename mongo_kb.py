import config
import llm_service
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import numpy as np
from knowledge_base_abc import KnowledgeBase
import time

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

class MongoKnowledgeBase(KnowledgeBase):
    """
    Manages the knowledge base stored in a local MongoDB collection.
    Similarity search is performed in-memory.
    """

    def __init__(self):
        """
        Initializes the connection to the MongoDB database and collection.
        """
        try:
            self.client = MongoClient(config.MONGO_URI)
            self.client.admin.command('ismaster')
            print("✅ Successfully connected to local MongoDB.")
        except ConnectionFailure as e:
            print(f"❌ Error: Could not connect to MongoDB.")
            print(f"Please ensure MongoDB is running and the URI in config.py is correct.")
            print(f"Details: {e}")
            self.client = None
            return

        self.db = self.client[config.DATABASE_NAME]
        self.collection = self.db[config.COLLECTION_NAME]

    def find_relevant_chunks(self, question_embedding, top_k=config.TOP_K):
        """
        Finds the most relevant text chunks by fetching all documents
        and calculating cosine similarity in-memory, with a recency bonus.
        """
        if not self.client or not question_embedding:
            return "", [], 0.0

        all_chunks = list(self.collection.find({}))
        if not all_chunks:
            return "", [], 0.0

        similarities = []
        now = time.time()
        for chunk in all_chunks:
            score = cosine_similarity(question_embedding, chunk["vector"])
            
            # Add a recency bonus to prioritize newly learned facts
            recency_bonus = 0.0
            created_at = chunk.get("created_at", 0.0) # Default to old if timestamp not present
            age_seconds = now - created_at
            
            # Increased bonus to 0.3 for newer items, decays over 7 days
            if age_seconds < 604800: # 7 days in seconds
                recency_bonus = 0.3 * (1 - (age_seconds / 604800))
            
            final_score = score + recency_bonus
            similarities.append((final_score, chunk))
    
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        top_score = similarities[0][0] if similarities else 0.0
        top_chunks = [chunk for score, chunk in similarities[:top_k] if score > 0]
        
        if not top_chunks:
            return "", [], 0.0

        context_str = "\n".join([chunk["content"] for chunk in top_chunks])
        sources = sorted(list(set([chunk["source"] for chunk in top_chunks])))
        
        return context_str, sources, top_score

    def learn_new_fact(self, fact_text, source="user_provided"):
        """
        Learns a new fact, adds it to the database with a timestamp.
        Returns True if successful, False otherwise.
        """
        if not self.client:
            print("❌ Cannot learn fact, no database connection.")
            return False

        print(f"🧠 Learning new fact: '{fact_text}'")
        new_embedding = llm_service.get_embedding(fact_text, config.EMBEDDING_MODEL)
        
        if new_embedding:
            self.collection.insert_one({
                "source": source,
                "content": fact_text,
                "vector": new_embedding,
                "created_at": time.time() # Add timestamp for new facts
            })
            print("✅ Fact learned successfully.")
            return True
        else:
            print("❌ Failed to create embedding. The fact was not learned.")
            return False

    def forget_fact(self, fact_text):
        """
        Finds and removes a fact from the database based on semantic similarity.
        """
        if not self.client:
            print("❌ Cannot forget fact, no database connection.")
            return

        print(f"🔎 Searching for a fact similar to: '{fact_text}' to forget...")

        fact_embedding = llm_service.get_embedding(fact_text, config.EMBEDDING_MODEL)
        if not fact_embedding:
            print("❌ Could not process the fact to forget. Please try rephrasing.")
            return
            
        all_chunks = list(self.collection.find({}))
        if not all_chunks:
            print("🤔 The knowledge base is empty. Nothing to forget.")
            return

        similarities = []
        for chunk in all_chunks:
            score = cosine_similarity(fact_embedding, chunk["vector"])
            similarities.append((score, chunk))

        similarities.sort(key=lambda x: x[0], reverse=True)
        best_match_score, best_match_chunk = similarities[0]

        if best_match_score >= config.FORGET_SIMILARITY_THRESHOLD:
            print(f"🎯 Found a fact in my knowledge base that seems to match your request (confidence {best_match_score:.2f}):")
            print(f"   '{best_match_chunk['content']}'")
            print(f"\n   This fact might be the cause of recent incorrect answers.")

            try:
                confirmation = input("   Are you sure you want me to permanently delete this specific fact? (yes/no): ").lower().strip()
                if confirmation == 'yes':
                    self.collection.delete_one({"_id": best_match_chunk['_id']})
                    print("✅ Fact forgotten successfully.")
                else:
                    print("👍 Deletion cancelled.")
            except (KeyboardInterrupt, EOFError):
                print("\n👍 Deletion cancelled.")

        else:
            print(f"🤷 I couldn't find a confident match. The best match had a score of {best_match_score:.2f}.")
            print(f"   My closest guess was: '{best_match_chunk['content']}'")
            print("   Nothing was deleted.")

    def handle_correction(self, last_question, correction_text, last_context=None):
        """
        Replaces a fact that was likely used to answer the last question with a correction.
        It prioritizes searching within the context of the last response if provided.
        """
        if not self.client:
            print("❌ Cannot handle correction, no database connection.")
            return
            
        print("✍️ It looks like you're correcting me.")
        print(f"   New information provided: '{correction_text}'")

        if not last_question:
            print("\n   I don't have the context of the last question. I'll learn this as new information.")
            self.learn_new_fact(correction_text)
            return

        question_embedding = llm_service.get_embedding(last_question, config.EMBEDDING_MODEL)
        if not question_embedding:
            print("\n   Could not process the last question. I'll learn this as new information.")
            self.learn_new_fact(correction_text)
            return

        # Determine the search space: prioritize the last context, fall back to the whole DB
        search_space = []
        if last_context:
            print("   (Using context from last answer to find fact to correct)")
            # We need the full document, not just the content string.
            # So we find the corresponding docs in the database.
            search_space = list(self.collection.find({"content": {"$in": last_context}}))
        
        if not search_space:
            print("   (No context provided or found, searching entire knowledge base)")
            search_space = list(self.collection.find({}))

        if not search_space:
            print("\n   The knowledge base is empty. Learning as new info.")
            self.learn_new_fact(correction_text)
            return

        similarities = []
        for chunk in search_space:
            score = cosine_similarity(question_embedding, chunk["vector"])
            similarities.append((score, chunk))

        similarities.sort(key=lambda x: x[0], reverse=True)
        
        if not similarities:
             print("\n   Could not find a relevant fact to correct. Learning as new info.")
             self.learn_new_fact(correction_text)
             return

        best_match_score, best_match_chunk = similarities[0]
        
        print(f"\n   I think this correction relates to the following fact in my knowledge:")
        print(f"   '{best_match_chunk['content']}'")
        print(f"\n   Should I replace this fact with the new information you provided?")
        
        try:
            confirmation = input("   (yes/no): ").lower().strip()
            if confirmation == 'yes':
                new_embedding = llm_service.get_embedding(correction_text, config.EMBEDDING_MODEL)
                if not new_embedding:
                    print("❌ I had trouble understanding the new info. Nothing was changed.")
                    return

                self.collection.update_one(
                    {"_id": best_match_chunk['_id']},
                    {"$set": {"content": correction_text, "vector": new_embedding}}
                )
                print("✅ Okay, I've updated my knowledge.")
            else:
                print("\n   Okay, I won't replace it. Would you like me to learn it as a completely new fact?")
                learn_as_new_confirmation = input("   (yes/no): ").lower().strip()
                if learn_as_new_confirmation == 'yes':
                    self.learn_new_fact(correction_text)
                else:
                    print("👍 Okay, correction cancelled.")
        except (KeyboardInterrupt, EOFError):
            print("\n👍 Correction cancelled.") 