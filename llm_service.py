import requests
import json
import config
import time

def get_embedding(text, model_name=config.EMBEDDING_MODEL):
    """Gets a vector embedding for text from Ollama."""
    try:
        payload = {"model": model_name, "prompt": text}
        response = requests.post(config.OLLAMA_API_URL_EMBEDDINGS, json=payload)
        response.raise_for_status()
        return response.json().get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Ollama for embeddings: {e}")
        return None

def generate_response(question, context, language, conversation_history, sources):
    """
    Sends the question, context, history, and sources to the LLM.
    Returns the generated response as a string.
    Enhanced with reasoning capabilities and more intelligent context handling.
    """
    history_str = format_history(conversation_history)
    
    # Enhanced prompt templates for better reasoning and more natural responses
    prompt_template_en = """
    You are BramAI, a highly intelligent and helpful assistant with a friendly, conversational personality.

    Follow these steps to answer the user's question:
    1. Carefully analyze the CONTEXT and conversation history.
    2. Connect relevant information from different parts of the CONTEXT if necessary.
    3. If the question requires inference beyond the explicit facts, reason through it step by step.
    4. Ensure your answer is comprehensive yet concise.
    5. Maintain a natural, conversational tone.

    If the answer is not contained in the CONTEXT or conversation history, respond with: "I'm sorry, I don't have that information."
    Do not mention sources in your response.

    RECENT CONVERSATION:
    {history}
    
    CONTEXT:
    {context}

    USER'S QUESTION:
    {question}
    """

    prompt_template_id = """
    Anda adalah BramAI, asisten yang sangat cerdas dan membantu dengan kepribadian yang ramah dan komunikatif.

    Ikuti langkah-langkah berikut untuk menjawab pertanyaan pengguna:
    1. Analisis dengan cermat KONTEKS dan riwayat percakapan.
    2. Hubungkan informasi yang relevan dari berbagai bagian KONTEKS jika diperlukan.
    3. Jika pertanyaan membutuhkan kesimpulan di luar fakta eksplisit, bernalar langkah demi langkah.
    4. Pastikan jawaban Anda komprehensif namun ringkas.
    5. Pertahankan nada percakapan yang alami.

    Jika jawaban tidak terdapat dalam KONTEKS atau riwayat percakapan, jawab dengan: "Maaf, saya tidak memiliki informasi tersebut."
    Jangan menyebutkan sumber informasi dalam jawaban Anda.

    RIWAYAT PERCAKAPAN TERBARU:
    {history}

    KONTEKS:
    {context}

    PERTANYAAN PENGGUNA:
    {question}
    """
    
    if language == 'id':
        prompt = prompt_template_id.format(history=history_str, context=context, question=question)
    else:
        prompt = prompt_template_en.format(history=history_str, context=context, question=question)
    
    try:
        # Enhanced parameters for better response quality
        payload = {
            "model": config.LLM_MODEL, 
            "prompt": prompt, 
            "stream": True,
            "temperature": 0.7,  # Balanced creativity and accuracy
            "top_k": 40,
            "top_p": 0.9
        }
        
        full_response = ""
        with requests.post(config.OLLAMA_API_URL_GENERATE, json=payload, stream=True) as response:
            response.raise_for_status()
            
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = json.loads(chunk.decode('utf-8'))
                    response_part = decoded_chunk.get("response", "")
                    full_response += response_part
                    if decoded_chunk.get("done"):
                        break
        return full_response.strip()

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return None

def summarize_text(text_to_summarize, language="id"):
    """
    Sends a long text to the LLM to be summarized.
    Streams the response back to the user.
    Now supports language parameter for proper summarization language.
    """
    print("📝 Meringkas teks..." if language == "id" else "📝 Summarizing the provided text...")

    # Enhanced templates for better summarization
    prompt_template_en = """
    You are a summarization expert. Provide a concise, well-structured summary 
    of the following text in English. Capture key points, main ideas, and important details.
    Make the summary cohesive and easy to understand.

    TEXT TO SUMMARIZE:
    {text}

    SUMMARY:
    """
    
    prompt_template_id = """
    Anda adalah ahli peringkasan. Berikan ringkasan yang singkat, terstruktur dengan baik
    dari teks berikut dalam Bahasa Indonesia. Tangkap poin utama, ide pokok, dan detail penting.
    Buat ringkasan yang kohesif dan mudah dipahami.

    TEKS YANG AKAN DIRINGKAS:
    {text}

    RINGKASAN:
    """
    
    prompt = prompt_template_id.format(text=text_to_summarize) if language == "id" else prompt_template_en.format(text=text_to_summarize)
    
    try:
        # Enhanced parameters for better summarization quality
        payload = {
            "model": config.LLM_MODEL, 
            "prompt": prompt, 
            "stream": True,
            "temperature": 0.5,  # More factual for summaries
            "top_k": 40,
            "top_p": 0.9
        }
        
        header = "\n🤖 Ringkasan BramAI: " if language == "id" else "\n🤖 BramAI Summary: "
        print(header, end="")
        
        with requests.post(config.OLLAMA_API_URL_GENERATE, json=payload, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = json.loads(chunk.decode('utf-8'))
                    response_part = decoded_chunk.get("response", "")
                    print(response_part, end="", flush=True)
                    if decoded_chunk.get("done"):
                        break
            print()

    except Exception as e:
        error_msg = f"\n❌ Terjadi kesalahan saat meringkas: {e}" if language == "id" else f"\n❌ An unexpected error occurred during summarization: {e}"
        print(error_msg)

def analyze_sentiment(text, language="id"):
    """
    Analyzes the sentiment of the provided text.
    Returns a sentiment score and explanation.
    """
    prompt_template_en = """
    Analyze the sentiment of the following text. Provide:
    1. A sentiment score from 1-10 (1 being very negative, 10 being very positive)
    2. A brief explanation of the sentiment

    TEXT TO ANALYZE:
    {text}

    RESULT:
    """
    
    prompt_template_id = """
    Analisis sentimen dari teks berikut. Berikan:
    1. Skor sentimen dari 1-10 (1 sangat negatif, 10 sangat positif)
    2. Penjelasan singkat mengenai sentimen tersebut

    TEKS YANG AKAN DIANALISIS:
    {text}

    HASIL:
    """
    
    prompt = prompt_template_id.format(text=text) if language == "id" else prompt_template_en.format(text=text)
    
    try:
        payload = {
            "model": config.LLM_MODEL, 
            "prompt": prompt
        }
        
        response = requests.post(config.OLLAMA_API_URL_GENERATE, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        error_msg = f"Terjadi kesalahan saat menganalisis sentimen: {e}" if language == "id" else f"An error occurred during sentiment analysis: {e}"
        return error_msg

def answer_with_reasoning(question, context, language="id"):
    """
    Provides an answer with explicit reasoning steps for complex questions.
    Shows the thought process to increase transparency and trust.
    """
    prompt_template_en = """
    You are a helpful AI assistant. For this complex question, show your reasoning process step by step:
    
    1. First, identify what is being asked and what information you need
    2. Break down the reasoning into clear, logical steps
    3. Connect relevant facts from the context
    4. Draw conclusions based on the evidence
    5. Provide your final answer
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    STEP-BY-STEP REASONING AND ANSWER:
    """
    
    prompt_template_id = """
    Anda adalah asisten AI yang membantu. Untuk pertanyaan kompleks ini, tunjukkan proses penalaran Anda langkah demi langkah:
    
    1. Pertama, identifikasi apa yang ditanyakan dan informasi apa yang Anda butuhkan
    2. Jabarkan penalaran menjadi langkah-langkah yang jelas dan logis
    3. Hubungkan fakta-fakta relevan dari konteks
    4. Tarik kesimpulan berdasarkan bukti
    5. Berikan jawaban akhir Anda
    
    KONTEKS:
    {context}
    
    PERTANYAAN:
    {question}
    
    PENALARAN LANGKAH DEMI LANGKAH DAN JAWABAN:
    """
    
    prompt = prompt_template_id.format(context=context, question=question) if language == "id" else prompt_template_en.format(context=context, question=question)
    
    try:
        payload = {"model": config.LLM_MODEL, "prompt": prompt}
        response = requests.post(config.OLLAMA_API_URL_GENERATE, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        error_msg = f"Terjadi kesalahan: {e}" if language == "id" else f"An error occurred: {e}"
        return error_msg

# Helper function moved here as it's coupled with the prompt generation
def format_history(history: list) -> str:
    """Formats the conversation history into a string for the prompt."""
    if not history:
        return "Tidak ada percakapan sebelumnya." if config.DEFAULT_LANGUAGE == "id" else "No recent conversation."
    
    formatted_string = ""
    for turn in history:
        user_prefix = "Pengguna" if turn.get("language", "") == "id" else "User"
        ai_prefix = "BramAI"
        formatted_string += f"{user_prefix}: {turn['user']}\n"
        formatted_string += f"{ai_prefix}: {turn['ai']}\n"
    return formatted_string.strip() 