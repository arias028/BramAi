from abc import ABC, abstractmethod

class KnowledgeBase(ABC):
    """
    Abstract Base Class for all knowledge base implementations.
    It defines the standard interface for interacting with a knowledge base.
    """

    @abstractmethod
    def find_relevant_chunks(self, question_embedding, **kwargs):
        """
        Finds the most relevant text chunks for a given question embedding.
        Must return a tuple of (context_string, list_of_sources, top_score).
        """
        pass

    @abstractmethod
    def learn_new_fact(self, fact_text, source="user_provided"):
        """
        Adds a new fact to the knowledge base.
        """
        pass

    @abstractmethod
    def forget_fact(self, fact_text):
        """
        Removes a fact from the knowledge base based on semantic similarity.
        """
        pass

    @abstractmethod
    def handle_correction(self, last_question, correction_text, last_context=None):
        """
        Corrects a fact in the knowledge base.
        It can use the context from the last response to be more precise.
        """
        pass 