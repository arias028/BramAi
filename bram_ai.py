import config
import llm_service
from mongo_kb import MongoKnowledgeBase
from knowledge_base_abc import KnowledgeBase
from langdetect import detect, LangDetectException
from flask import Flask, request, jsonify
import threading
import time
import sys
import argparse
import re
from web_search import ddg_search # Import the renamed web search function

GREETINGS = [
    "halo", "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "pagi", "siang", "sore", "malam", "oi", "hi", "hello", "hey"
]

class BramAI:
    """
    The core class for the BramAI assistant.
    Encapsulates the application logic, including conversation management,
    command processing, interaction with the knowledge base, and advanced reasoning.
    """
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.conversation_history = []
        self.last_user_input = ""
        # Enhanced response details with more context tracking
        self.last_response_details = {
            "text": "", 
            "context_chunks": [], 
            "language": "id", 
            "timestamp": time.time(),
            "understanding_score": 0.0
        }

    def _detect_language(self, text: str) -> str:
        """
        Detects the language of the given text using langdetect.
        Now defaults to 'id' (Indonesian) unless explicitly detected as English.
        """
        try:
            # More robust language detection for short texts
            detected = detect(text)
            if detected == 'en':
                return 'en'
            # Default to Indonesian for any non-English language
            return 'id'
        except LangDetectException:
            # If detection fails, default to Indonesian
            return 'id'

    def web_search_and_respond(self, query: str, language: str) -> str:
        """
        Handles the web search, response generation, and self-learning process.
        """
        try:
            # Ensure the query is clean user input for the highest quality search
            search_results = ddg_search(query=query, max_results=5)
            
            if not search_results or not search_results.results:
                print("❌ Tidak ada hasil yang ditemukan dari internet.")
                return "Maaf, saya tidak dapat menemukan informasi dari internet saat ini."
            
            print(f"✅ Ditemukan {len(search_results.results)} hasil dari internet. Merangkum jawaban...")
            web_context = " ".join([result.body for result in search_results.results])
            
            ai_response = llm_service.generate_response_from_web(query, web_context, language)
            
            if ai_response and "tidak dapat menemukan jawaban" not in ai_response:
                print("🧠 Mempelajari informasi baru dari internet...")
                learned_fact = f"Ketika ditanya '{query}', jawabannya adalah: {ai_response}"
                self.kb.learn_new_fact(learned_fact, source=f"web_search: {search_results.results[0].url}")
                return ai_response
            else:
                return "Saya menemukan beberapa informasi, tetapi kesulitan untuk merangkum jawaban yang jelas."

        except Exception as e:
            print(f"❌ Terjadi kesalahan saat mencari di web: {e}")
            return "Maaf, saya mengalami masalah saat mencoba mencari informasi di internet."

    def process_input(self, user_input: str) -> str:
        """
        Processes the user's input, whether it's a command or a question,
        and returns the AI's response as a string. Enhanced with better context
        handling and more intelligent responses.
        """
        if not user_input or not user_input.strip():
            return "Mohon ketikkan pertanyaan atau perintah Anda."
        
        response_str = ""
        language = self._detect_language(user_input)
        self.last_response_details["language"] = language
        
        # ---> BLOK KRITIS BARU UNTUK COMMAND OVERRIDE <---
        search_keywords = ["cari di internet", "carikan saya", "search for", "find me", "tolong carikan"]
        user_input_lower = user_input.lower()

        for keyword in search_keywords:
            if keyword in user_input_lower:
                # Ekstrak query yang bersih dari perintah
                search_query = re.sub(keyword, '', user_input, flags=re.IGNORECASE).strip()
                print(f"🔍 Perintah pencarian eksplisit terdeteksi. Mencari: '{search_query}'")
                
                # Jika query kosong setelah dihapus (misal: "tolong carikan"), gunakan input sebelumnya
                if not search_query:
                    search_query = self.last_user_input
                
                return self.web_search_and_respond(search_query, language)
        # ---> AKHIR BLOK KRITIS <---

        # Initialize variables that might be used later
        relevant_context = ""
        top_score = 0.0
        sources = []
        
        # Step 1: Handle simple greetings first for a fast response
        # More flexible greeting detection - check if ANY greeting word is in the input
        any_greeting_word_found = any(word in user_input.lower().split() for word in GREETINGS)
        if any_greeting_word_found:
            response_str = "Halo! Ada yang bisa saya bantu?" if language == "id" else "Hello! How can I help you?"
            self.last_response_details["text"] = response_str
            return response_str

        # The rest of the original logic for checking local KB
        self.last_user_input = user_input
        question_embedding = llm_service.get_embedding(user_input, config.EMBEDDING_MODEL)
            
        if question_embedding is None:
            response_str = "Maaf, saya sedang kesulitan terhubung ke layanan inti saya. Pastikan Ollama sudah berjalan." if language == "id" else "I'm sorry, I'm having trouble connecting to my core services. Please make sure Ollama is running."
            # Early exit if embedding service fails
            self.last_response_details["text"] = response_str
            return response_str

        # Step 1: Check local knowledge base first
        chunk_result = self.kb.find_relevant_chunks(question_embedding, top_k=config.TOP_K) or ("", [], 0.0)
        relevant_context, sources, top_score = chunk_result
            
        # Step 2: Decide if local knowledge is sufficient or if web search is needed
        if top_score < config.WEB_SEARCH_THRESHOLD:
            # JIKA SKOR RENDAH, LANGSUNG CARI DI INTERNET
            print(f"⚠️ Pengetahuan lokal kurang memadai (skor: {top_score:.2f}). Mencari di internet...")
            ai_response = self.web_search_and_respond(user_input, language)
        else:
            # JIKA SKOR TINGGI, GUNAKAN BASIS DATA LOKAL
            print("✅ Menjawab dari basis data pengetahuan lokal (Percobaan Pertama).")
            ai_response = llm_service.generate_response(
                user_input, 
                relevant_context, 
                language, 
                self.conversation_history, 
                sources
            )
            
            # NEW FALLBACK: Check if the AI itself couldn't find the answer
            if ai_response and "informasi tersebut tidak ada di basis data saya" in ai_response:
                print("⚠️ Jawaban tidak ditemukan di konteks lokal. Memulai pencarian web sebagai fallback...")
                ai_response = self.web_search_and_respond(user_input, language)
        
        if ai_response:
            response_str = ai_response
        else:
            response_str = "Maaf, saya tidak dapat menemukan atau menghasilkan jawaban untuk pertanyaan Anda saat ini." if language == "id" else "I am unable to generate a response at this moment."
        
        # Store detailed response info for better correction handling
        self.last_response_details = {
            "text": response_str,
            "context_chunks": relevant_context.split('\n') if relevant_context else [],
            "language": language,
            "timestamp": time.time(),
            "understanding_score": top_score
        }
        return response_str


def run_terminal_chat(ai):
    """
    Run an interactive terminal chat with BramAI.
    """
    language = config.DEFAULT_LANGUAGE
    welcome_msg = "Selamat datang di BramAI! Ketik 'keluar' untuk keluar." if language == "id" else "Welcome to BramAI! Type 'exit' to quit."
    print(f"\n🤖 {welcome_msg}\n")
    
    while True:
        try:
            # Get input with nice prompt based on language
            prompt = "Anda: " if language == "id" else "You: "
            user_input = input(prompt)
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "keluar", "selesai"]:
                goodbye_msg = "Terima kasih telah menggunakan BramAI! Sampai jumpa." if language == "id" else "Thank you for using BramAI! Goodbye."
                print(f"🤖 {goodbye_msg}")
                break
            
            # Process the input and get the response
            response = ai.process_input(user_input)
            
            # Update language for next prompt
            language = ai.last_response_details.get("language", config.DEFAULT_LANGUAGE)
            
            # Print the response
            ai_prefix = "BramAI: "
            print(f"{ai_prefix}{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting BramAI terminal chat...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


app = Flask(__name__)

# Initialize the AI just once
kb = MongoKnowledgeBase()
if not hasattr(kb, 'client') or not kb.client:
    print("⚠️  Could not connect to the knowledge base. Exiting.")
    exit()
    
ai = BramAI(knowledge_base=kb)

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

@app.route('/propose_correction', methods=['POST'])
def propose_correction_endpoint():
    data = request.json
    if data is None:
        return jsonify({'error': 'Request must be a JSON object.'}), 400
    last_question = data.get('last_question')
    if not last_question:
        return jsonify({'error': 'last_question is required'}), 400

    proposal = ai.kb.propose_correction_and_get_original(last_question)
    if proposal:
        return jsonify(proposal)
    else:
        return jsonify({'error': 'Could not find a relevant fact to correct'}), 404

@app.route('/confirm_correction', methods=['POST'])
def confirm_correction_endpoint():
    data = request.json
    if data is None:
        return jsonify({'error': 'Request must be a JSON object.'}), 400
    document_id = data.get('document_id')
    new_text = data.get('new_text')
    if not all([document_id, new_text]):
        return jsonify({'error': 'document_id and new_text are required'}), 400

    success = ai.kb.confirm_correction(document_id, new_text)
    if success:
        return jsonify({'status': 'success', 'message': 'Knowledge base updated.'})
    else:
        return jsonify({'error': 'Failed to update the fact in the database'}), 500

@app.route('/propose_forget', methods=['POST'])
def propose_forget_endpoint():
    data = request.json
    if data is None:
        return jsonify({'error': 'Request must be a JSON object.'}), 400
    fact_text = data.get('fact_text')
    if not fact_text:
        return jsonify({'error': 'fact_text is required'}), 400

    proposal = ai.kb.propose_fact_to_forget(fact_text)
    if proposal:
        return jsonify(proposal)
    else:
        return jsonify({'error': 'Could not find a relevant fact to forget'}), 404

@app.route('/confirm_forget', methods=['POST'])
def confirm_forget_endpoint():
    data = request.json
    if data is None:
        return jsonify({'error': 'Request must be a JSON object.'}), 400
    document_id = data.get('document_id')
    if not document_id:
        return jsonify({'error': 'document_id is required'}), 400

    success = ai.kb.confirm_forget(document_id)
    if success:
        return jsonify({'status': 'success', 'message': 'Fact has been forgotten.'})
    else:
        return jsonify({'error': 'Failed to delete the fact from the database'}), 500

def run_flask_app():
    # You can choose a different port if 5000 is in use
    app.run(host='0.0.0.0', port=5001)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BramAI - Super Indonesian AI Assistant")
    parser.add_argument("--web", action="store_true", help="Run as web service with Flask")
    parser.add_argument("--terminal", action="store_true", help="Run in terminal mode")
    args = parser.parse_args()
    
    # Default to terminal mode if no arguments provided
    if not args.web and not args.terminal:
        args.terminal = True
        
    if args.web:
        print("🤖 BramAI siap menerima permintaan web.")
        # Run Flask in a separate thread so it doesn't block other code if needed
        flask_thread = threading.Thread(target=run_flask_app)
        flask_thread.start()
    
    if args.terminal:
        run_terminal_chat(ai)
