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
    def forget_fact(self, fact_text):
        """
        Removes a fact from the knowledge base based on semantic similarity.
        
        Parameters:
        - fact_text: Text description of the fact to forget
        """
        pass

    @abstractmethod
    def handle_correction(self, last_question, correction_text, last_context=None):
        """
        Corrects a fact in the knowledge base.
        It can use the context from the last response to be more precise.
        
        Parameters:
        - last_question: The previous question that triggered the incorrect answer
        - correction_text: The correction text provided by the user
        - last_context: The context used for the last response (optional)
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