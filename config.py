# The MONGO_URI is now updated to your new MongoDB Atlas connection string.
# Make sure this is the only line you change here.
MONGO_URI = "mongodb+srv://bramteknologi:jhWg8zK70vxLpJgJ@bramai.ivqn9h3.mongodb.net/?retryWrites=true&w=majority&appName=BramAi"

DATABASE_NAME = "BramAi"
COLLECTION_NAME = "knowledge"
OLLAMA_API_URL_GENERATE = "http://127.0.0.1:11434/api/generate"
OLLAMA_API_URL_EMBEDDINGS = "http://127.0.0.1:11434/api/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3:8b"
TOP_K = 8  # Increased from 5 to get more context
CONVERSATION_HISTORY_LENGTH = 5  # Increased from 3 for better context awareness
FORGET_SIMILARITY_THRESHOLD = 0.9
CLARIFICATION_THRESHOLD = 0.5
WEB_SEARCH_THRESHOLD = 0.82  # Increased from 0.7 to be stricter
DEFAULT_LANGUAGE = "id"  # Set Indonesian as default language
TEMPERATURE = 0.7  # Default creativity level
REASONING_THRESHOLD = 0.75  # Threshold to use advanced reasoning for complex questions
