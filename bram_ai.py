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
        
        # Initialize variables that might be used later
        relevant_context = ""
        top_score = 0.0
        sources = []
        
        # Step 1: Handle simple greetings first for a fast response
        cleaned_input = re.sub(r'[^\w\s]', '', user_input.lower().strip())
        if cleaned_input in GREETINGS:
            response_str = "Halo! Ada yang bisa saya bantu?" if language == "id" else "Hello! How can I help you?"
            self.last_response_details["text"] = response_str
            return response_str

        if user_input.lower().startswith(("learn:", "pelajari:")):
            fact_to_learn = user_input.split(":", 1)[1].strip()
            if fact_to_learn:
                if self.kb.learn_new_fact(fact_to_learn):
                    response_str = "✅ Saya sudah mempelajari informasi baru ini." if language == "id" else "✅ Understood! I've learned that new information."
                else:
                    response_str = "❌ Maaf, saya kesulitan mempelajari itu. Mungkin masalah dengan layanan embedding." if language == "id" else "❌ I'm sorry, I had trouble learning that. It might be an issue with the embedding service."
            else:
                response_str = "🤖 Mohon berikan informasi yang ingin saya pelajari setelah 'pelajari:' atau 'learn:'." if language == "id" else "🤖 Please provide the information you want me to learn after 'learn:'."
        
        elif user_input.lower().startswith(("forget:", "lupakan:")):
            fact_to_forget = user_input.split(":", 1)[1].strip()
            if fact_to_forget:
                self.kb.forget_fact(fact_to_forget)
                response_str = "🔎 Sedang memproses permintaan untuk melupakan informasi." if language == "id" else "🔎 Processing your request to forget."
            else:
                response_str = "🤖 Mohon berikan informasi yang ingin saya lupakan setelah 'lupakan:' atau 'forget:'." if language == "id" else "🤖 Please provide the information you want me to forget after 'forget:'."

        elif user_input.lower().startswith(("koreksi:", "correct:")):
            fact_to_correct = user_input.split(":", 1)[1].strip()
            if fact_to_correct:
                # Enhanced correction with last context and understanding scoring
                last_context = self.last_response_details.get("context_chunks", [])
                self.kb.handle_correction(self.last_user_input, fact_to_correct, last_context)
                response_str = "✍️ Sedang memproses koreksi berdasarkan konteks terakhir." if language == "id" else "✍️ Processing your correction based on the last context."
            else:
                response_str = "🤖 Mohon berikan informasi yang benar setelah 'koreksi:' atau 'correct:'." if language == "id" else "🤖 Please provide the correct information after 'correct:'."
        
        elif user_input.lower().startswith(("ringkas:", "summarize:")):
            text_to_summarize = user_input.split(":", 1)[1].strip()
            if text_to_summarize:
                # Pass language to summarize in the right language
                llm_service.summarize_text(text_to_summarize, language)
                response_str = "Ringkasan sedang dibuat." if language == "id" else "Summary is being generated."
            else:
                response_str = "🤖 Mohon berikan teks yang ingin saya ringkas setelah 'ringkas:' atau 'summarize:'." if language == "id" else "🤖 Please provide the text you want me to summarize after 'summarize:'."

        elif user_input.lower().startswith(("analisis:", "analyze:")):
            text_to_analyze = user_input.split(":", 1)[1].strip()
            if text_to_analyze:
                response_str = llm_service.analyze_sentiment(text_to_analyze, language)
            else:
                response_str = "🤖 Mohon berikan teks yang ingin saya analisis setelah 'analisis:' atau 'analyze:'." if language == "id" else "🤖 Please provide the text you want me to analyze after 'analyze:'."

        else: # It's a regular question
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
                try:
                    # Ensure the query is clean user input
                    search_results = ddg_search(query=self.last_user_input, max_results=5)
                    
                    if not search_results or not search_results.results:
                        print("❌ Tidak ada hasil yang ditemukan dari internet.")
                        ai_response = "Maaf, saya tidak dapat menemukan informasi dari internet saat ini."
                    else:
                        print(f"✅ Ditemukan {len(search_results.results)} hasil dari internet. Merangkum jawaban...")
                        web_context = " ".join([result.body for result in search_results.results])
                        
                        ai_response = llm_service.generate_response_from_web(user_input, web_context, language)
                        
                        if ai_response and "tidak dapat menemukan jawaban" not in ai_response:
                            # Self-learning: learn the new fact from the web
                            print("🧠 Mempelajari informasi baru dari internet...")
                            learned_fact = f"Ketika ditanya '{user_input}', jawabannya adalah: {ai_response}"
                            self.kb.learn_new_fact(learned_fact, source=f"web_search: {search_results.results[0].url}")
                        else:
                             ai_response = "Saya menemukan beberapa informasi, tetapi kesulitan untuk merangkum jawaban yang jelas."

                except Exception as e:
                    print(f"❌ Terjadi kesalahan saat mencari di web: {e}")
                    ai_response = "Maaf, saya mengalami masalah saat mencoba mencari informasi di internet."
            else:
                # JIKA SKOR TINGGI, GUNAKAN BASIS DATA LOKAL
                print("✅ Menjawab dari basis data pengetahuan lokal.")
                ai_response = llm_service.generate_response(
                    user_input, 
                    relevant_context, 
                    language, 
                    self.conversation_history, 
                    sources
                )

            if ai_response:
                response_str = ai_response
                # Store enhanced context for better conversation tracking
                self.conversation_history.append({
                    "user": user_input, 
                    "ai": ai_response,
                    "context": relevant_context,
                    "language": language,
                    "timestamp": time.time(),
                    "understanding_score": top_score
                })
                if len(self.conversation_history) > config.CONVERSATION_HISTORY_LENGTH:
                    self.conversation_history.pop(0)
            else:
                response_str = "Maaf, saya tidak bisa menghasilkan jawaban saat ini." if language == "id" else "I am unable to generate a response at this moment."
        
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
