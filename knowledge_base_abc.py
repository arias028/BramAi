from abc import ABC, abstractmethod

class KnowledgeBase(ABC):
    """
    Abstract Base Class for all knowledge base implementations.
    It defines the standard interface for interacting with a knowledge base.
    Enhanced with additional methods for advanced AI capabilities.
    """

    @abstractmethod
    def find_relevant_chunks(self, question_embedding, top_k=5, **kwargs):
        """
        Finds the most relevant text chunks for a given question embedding.
        Must return a tuple of (context_string, list_of_sources, top_score).
        
        Parameters:
        - question_embedding: The vector embedding of the question
        - top_k: Number of chunks to retrieve (default: 5)
        - kwargs: Additional parameters for specific implementations
        """
        pass

    @abstractmethod
    def learn_new_fact(self, fact_text, source="user_provided", metadata=None):
        """
        Adds a new fact to the knowledge base.
        
        Parameters:
        - fact_text: The text of the fact to add
        - source: The source of the fact (default: "user_provided")
        - metadata: Additional metadata about the fact (optional)
        """
        pass

    @abstractmethod
    def propose_fact_to_forget(self, fact_text_to_find: str):
        """
        Finds a fact similar to the provided text and returns its details for confirmation.
        
        Parameters:
        - fact_text_to_find: Text description of the fact to find
        
        Returns:
        - Dictionary with document_id and content_to_forget, or None if not found
        """
        pass

    @abstractmethod
    def confirm_forget(self, document_id: str):
        """
        Removes a fact from the knowledge base by its document ID after confirmation.
        
        Parameters:
        - document_id: The ID of the document to delete
        
        Returns:
        - Boolean indicating success or failure
        """
        pass

    @abstractmethod
    def propose_correction_and_get_original(self, last_question: str):
        """
        Finds a fact related to the last question and returns its details for correction.
        
        Parameters:
        - last_question: The previous question that triggered the incorrect answer
        
        Returns:
        - Dictionary with document_id and original_content, or None if not found
        """
        pass
        
    @abstractmethod
    def confirm_correction(self, document_id: str, new_fact_text: str):
        """
        Updates a fact in the knowledge base by its document ID after confirmation.
        
        Parameters:
        - document_id: The ID of the document to update
        - new_fact_text: The corrected text to replace the original fact
        
        Returns:
        - Boolean indicating success or failure
        """
        pass
        
    def find_related_facts(self, topic, limit=10):
        """
        Finds facts related to a specific topic.
        Optional method that implementations can override.
        
        Parameters:
        - topic: The topic to find related facts about
        - limit: Maximum number of facts to return
        
        Returns:
        - List of related facts with their metadata
        """
        return []
        
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