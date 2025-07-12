import config
import llm_service
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
# Import the new ServerApi class
from pymongo.server_api import ServerApi
import numpy as np
from knowledge_base_abc import KnowledgeBase
import time
from bson.objectid import ObjectId

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
    Manages the knowledge base stored in a MongoDB Atlas collection.
    Similarity search is now handled by MongoDB's optimized Vector Search.
    """

    def __init__(self):
        """
        Initializes the connection to the MongoDB Atlas database and collection
        using the modern, stable API connection method.
        """
        try:
            # Create a new client and connect to the server using the URI from config
            # We now include server_api=ServerApi('1') for stable API versioning
            self.client = MongoClient(config.MONGO_URI, server_api=ServerApi('1'))
            
            # Send a ping to confirm a successful connection
            self.client.admin.command('ping')
            print("✅ Successfully connected to MongoDB Atlas!")

        except ConnectionFailure as e:
            print(f"❌ Error: Could not connect to MongoDB Atlas.")
            print(f"Please ensure your IP address is whitelisted in Atlas and the URI in config.py is correct.")
            print(f"Details: {e}")
            self.client = None
            return
        except Exception as e:
            print(f"❌ An unexpected error occurred during connection: {e}")
            self.client = None
            return

        self.db = self.client[config.DATABASE_NAME]
        self.collection = self.db[config.COLLECTION_NAME]

    def find_relevant_chunks(self, question_embedding, top_k=config.TOP_K):
        """
        Finds the most relevant text chunks using MongoDB Atlas Vector Search.
        This is much faster and more scalable than the old method.
        """
        if not self.client or not question_embedding:
            return "", [], 0.0

        # This is the new, efficient way to search using a vector index.
        # IMPORTANT: You must create a Vector Search Index in your Atlas UI named 'vector_index'
        # for this to work.
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'vector_index', # The name of your index in Atlas
                    'path': 'vector',
                    'queryVector': question_embedding,
                    'numCandidates': 150, # Number of candidates to consider
                    'limit': top_k
                }
            },
            {
                '$project': {
                    'content': 1,
                    'source': 1,
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            if not results:
                return "", [], 0.0
        except Exception as e:
            print(f"❌ Vector search failed. Is your Atlas index named 'vector_index'?")
            print(f"   Error details: {e}")
            return "", [], 0.0

        top_score = results[0]['score'] if results else 0.0
        
        # Filter results by a minimum score threshold to improve relevance
        # A score of 0.8 is a good starting point for cosine similarity
        relevant_chunks = [chunk for chunk in results if chunk['score'] > 0.8]

        if not relevant_chunks:
            return "", [], 0.0

        context_str = "\n".join([chunk["content"] for chunk in relevant_chunks])
        sources = sorted(list(set([chunk["source"] for chunk in relevant_chunks])))
        
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

    # Method to find a fact to correct and return its details for confirmation
    def propose_correction_and_get_original(self, last_question: str):
        if not self.client or not last_question:
            return None
        question_embedding = llm_service.get_embedding(last_question, config.EMBEDDING_MODEL)
        if not question_embedding:
            return None
        pipeline = [
            {'$vectorSearch': {'index': 'vector_index', 'path': 'vector', 'queryVector': question_embedding, 'numCandidates': 10, 'limit': 1}},
            {'$project': {'content': 1, 'source': 1}}
        ]
        try:
            results = list(self.collection.aggregate(pipeline))
            if not results:
                return None
            # Return the document ID and its content
            return {"document_id": str(results[0]['_id']), "original_content": results[0]['content']}
        except Exception:
            return None

    # Method to perform the update after user confirmation
    def confirm_correction(self, document_id: str, new_fact_text: str):
        if not self.client:
            return False
        new_embedding = llm_service.get_embedding(new_fact_text, config.EMBEDDING_MODEL)
        if not new_embedding:
            return False
        try:
            result = self.collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": {"content": new_fact_text, "vector": new_embedding, "updated_at": time.time()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error confirming correction: {e}")
            return False

    # Method to find a fact to forget and return its details for confirmation
    def propose_fact_to_forget(self, fact_text_to_find: str):
        if not self.client:
            return None
        fact_embedding = llm_service.get_embedding(fact_text_to_find, config.EMBEDDING_MODEL)
        if not fact_embedding:
            return None
        pipeline = [
            {'$vectorSearch': {'index': 'vector_index', 'path': 'vector', 'queryVector': fact_embedding, 'numCandidates': 10, 'limit': 1}},
            {'$project': {'content': 1, 'source': 1}}
        ]
        try:
            results = list(self.collection.aggregate(pipeline))
            if not results:
                return None
            return {"document_id": str(results[0]['_id']), "content_to_forget": results[0]['content']}
        except Exception:
            return None

    # Method to perform the deletion after user confirmation
    def confirm_forget(self, document_id: str):
        if not self.client:
            return False
        try:
            result = self.collection.delete_one({"_id": ObjectId(document_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Error confirming forget: {e}")
            return False

    def get_statistics(self):
        """
        Returns statistics about the knowledge base.
        Optional method that implementations can override.
        
        Returns:
        - Dictionary with statistics about the knowledge base
        """
        return {
            "total_facts": 0,
            "last_updated": None
        }