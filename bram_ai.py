import config
import llm_service
from mongo_kb import MongoKnowledgeBase
from knowledge_base_abc import KnowledgeBase
from langdetect import detect, LangDetectException
from flask import Flask, request, jsonify
import threading

class BramAI:
    """
    The core class for the BramAI assistant.
    Encapsulates the application logic, including conversation management,
    command processing, and interaction with the knowledge base.
    """
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.conversation_history = []
        self.last_user_input = ""
        # We will now store a dictionary for the last response to hold more data
        self.last_response_details = {"text": "", "context_chunks": []}

    def _detect_language(self, text: str) -> str:
        """
        Detects the language of the given text using langdetect.
        Defaults to 'en' if detection fails or the text is too short.
        """
        try:
            # The langdetect library is not deterministic.
            # For short texts, it might give different results each time.
            # We check for 'id' and default to 'en' for everything else.
            if detect(text) == 'id':
                return 'id'
            return 'en'
        except LangDetectException:
            # If the text is too short or ambiguous, it might fail.
            # Defaulting to English is a safe bet.
            return 'en'

    def process_input(self, user_input: str) -> str:
        """
        Processes the user's input, whether it's a command or a question,
        and returns the AI's response as a string.
        """
        response_str = ""
        
        if user_input.lower().startswith("learn:"):
            fact_to_learn = user_input[len("learn:"):].strip()
            if fact_to_learn:
                if self.kb.learn_new_fact(fact_to_learn):
                    response_str = "✅ Understood! I've learned that new information."
                else:
                    response_str = "❌ I'm sorry, I had trouble learning that. It might be an issue with creating the embedding. Please ensure the embedding service is running."
            else:
                response_str = "🤖 Please provide the information you want me to learn after 'learn:'."
        
        elif user_input.lower().startswith("forget:"):
            fact_to_forget = user_input[len("forget:"):].strip()
            if fact_to_forget:
                # This function now needs to be adapted to return string output
                # For now, we'll assume it prints to console and return a generic message.
                self.kb.forget_fact(fact_to_forget)
                response_str = "🔎 Processing your request to forget."
            else:
                response_str = "🤖 Please provide the information you want me to forget after 'forget:'."

        elif user_input.lower().startswith("koreksi:"):
            fact_to_correct = user_input[len("koreksi:"):].strip()
            if fact_to_correct:
                # Use the context from the last response for a more accurate correction
                last_context = self.last_response_details.get("context_chunks", [])
                self.kb.handle_correction(self.last_user_input, fact_to_correct, last_context)
                response_str = "✍️ Processing your correction based on the last context."
            else:
                response_str = "🤖 Please provide the correct information after 'koreksi:'."
        
        elif user_input.lower().startswith("ringkas:"):
            text_to_summarize = user_input[len("ringkas:"):].strip()
            if text_to_summarize:
                # Assuming summarize_text prints the output.
                llm_service.summarize_text(text_to_summarize)
                response_str = "Summary is being generated above."
            else:
                response_str = "🤖 Please provide the text you want me to summarize after 'ringkas:'."

        else: # It's a regular question
            self.last_user_input = user_input
            language = self._detect_language(user_input)
            question_embedding = llm_service.get_embedding(user_input, config.EMBEDDING_MODEL)
            
            # Ensure we have a tuple to unpack, even if the method were to return None.
            chunk_result = self.kb.find_relevant_chunks(question_embedding) or ("", [], 0.0)
            relevant_context, sources, top_score = chunk_result
            
            if top_score < config.CLARIFICATION_THRESHOLD:
                response_str = "I'm not quite sure what you mean. Could you try rephrasing the question?"
            else:
                ai_response = llm_service.generate_response(
                    user_input, 
                    relevant_context, 
                    language, 
                    self.conversation_history, 
                    sources
                )

                if ai_response:
                    response_str = ai_response
                    # Store context along with the conversation
                    self.conversation_history.append({
                        "user": user_input, 
                        "ai": ai_response,
                        "context": relevant_context
                    })
                    if len(self.conversation_history) > config.CONVERSATION_HISTORY_LENGTH:
                        self.conversation_history.pop(0)
                else:
                    response_str = "I am unable to generate a response at this moment."
        
        # Store details of this response for potential correction
        self.last_response_details = {
            "text": response_str,
            "context_chunks": relevant_context.split('\n') if 'relevant_context' in locals() else []
        }
        return response_str


app = Flask(__name__)

# Initialize the AI just once
kb = MongoKnowledgeBase()
if not hasattr(kb, 'client') or not kb.client:
    print("⚠️  Could not connect to the knowledge base. Exiting.")
    exit()
    
ai = BramAI(knowledge_base=kb)
print("🤖 BramAI is ready to receive web requests.")

@app.route('/webhook', methods=['POST'])
def handle_message():
    data = request.json
    if data is None:
        return jsonify({'error': 'Request must be a JSON object.'}), 400
        
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Process the input using your AI's logic
    ai_response = ai.process_input(user_message)

    # Return the AI's response
    return jsonify({'reply': ai_response})

def run_flask_app():
    # You can choose a different port if 5000 is in use
    app.run(host='0.0.0.0', port=5001)

if __name__ == "__main__":
    # Run Flask in a separate thread so it doesn't block other code if needed
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()
    # The original main loop can be removed or repurposed if you only want to use the web interface.
