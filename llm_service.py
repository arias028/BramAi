import requests
import json
import config

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
    """
    history_str = format_history(conversation_history)
    
    # This function depends on format_history, which is a UI-level helper.
    # For now, we'll need to pass it in or move it here. Let's move it.
    
    prompt_template_en = """
    You are BramAI, a friendly and helpful assistant. Your personality is casual.
    Remember the recent conversation history and use it for context if the user's question is a follow-up.
    Using ONLY the following CONTEXT and the conversation history, answer the user's question concisely in English.
    If the answer is not in the CONTEXT or history, you must say "I'm sorry, I don't have that information."
    Do not mention the source of your information in the answer.

    RECENT CONVERSATION:
    {history}
    
    CONTEXT:
    {context}

    USER'S QUESTION:
    {question}
    """

    prompt_template_id = """
    Anda adalah BramAI, asisten yang ramah dan suka membantu. Kepribadian Anda santai.
    Ingatlah riwayat percakapan terkini dan gunakan sebagai konteks jika pertanyaan pengguna adalah pertanyaan lanjutan.
    Gunakan HANYA informasi berikut (disebut "KONTEKS") dan riwayat percakapan untuk menjawab pertanyaan pengguna secara ringkas dalam Bahasa Indonesia.
    Jika jawaban tidak ditemukan di dalam KONTEKS atau riwayat, Anda harus menjawab "Maaf, saya tidak memiliki informasi mengenai hal itu."
    Jangan sebutkan sumber informasi dalam jawaban Anda.

    RIWAYAT PERCAKAPAN:
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
        payload = {"model": config.LLM_MODEL, "prompt": prompt, "stream": True}
        
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

def summarize_text(text_to_summarize):
    """
    Sends a long text to the LLM to be summarized.
    Streams the response back to the user.
    """
    print("📝 Summarizing the provided text...")

    prompt_template = """
    You are a summarization expert. Your task is to provide a concise, easy-to-read summary 
    of the following text in the same language as the original text. Capture the key points and main ideas.

    TEXT TO SUMMARIZE:
    {text}

    SUMMARY:
    """
    
    prompt = prompt_template.format(text=text_to_summarize)
    
    try:
        payload = {"model": config.LLM_MODEL, "prompt": prompt, "stream": True}
        
        print("\n🤖 BramAI Summary: ", end="")
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
        print(f"\n❌ An unexpected error occurred during summarization: {e}")

# Helper function moved here as it's coupled with the prompt generation
def format_history(history: list) -> str:
    """Formats the conversation history into a string for the prompt."""
    if not history:
        return "No recent conversation."
    
    formatted_string = ""
    for turn in history:
        formatted_string += f"User: {turn['user']}\n"
        formatted_string += f"BramAI: {turn['ai']}\n"
    return formatted_string.strip() 