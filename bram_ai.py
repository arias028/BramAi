import config
import llm_service
import knowledge_base

def detect_language(question: str) -> str:
    """
    A simple heuristic-based function to detect if the question is Indonesian.
    Defaults to English if no Indonesian keywords are found.
    """
    indonesian_keywords = [
        'siapa', 'apa', 'kapan', 'dimana', 'bagaimana', 'mengapa', 
        'jelaskan', 'beri tahu', 'berapa', 'ringkas'
    ]
    if any(keyword in question.lower() for keyword in indonesian_keywords):
        return 'id'
    return 'en'

def main():
    """Main interactive loop for BramAI."""
    print("🤖 BramAI (Refactored Edition) is online! Type 'exit' to quit.")
    
    vector_database = knowledge_base.load_vector_database()
    if not vector_database:
        print("⚠️  Could not load knowledge base. Exiting.")
        return
        
    print("You can now ask questions or use commands: 'learn:', 'koreksi:', 'forget:', or 'ringkas: [text]'")
    
    conversation_history = []

    while True:
        try:
            user_input = input("\n👤 You: ")
        except (KeyboardInterrupt, EOFError):
            print("\n🤖 BramAI signing off. Goodbye, Onii-chan!")
            break

        if user_input.lower() == 'exit':
            print("🤖 BramAI signing off. Goodbye, Onii-chan!")
            break
            
        if user_input.lower().startswith("learn:"):
            fact_to_learn = user_input[len("learn:"):].strip()
            if fact_to_learn:
                knowledge_base.learn_new_fact(fact_to_learn, vector_database)
            else:
                print("🤖 Please provide the information you want me to learn after 'learn:'.")
        
        elif user_input.lower().startswith("forget:"):
            fact_to_forget = user_input[len("forget:"):].strip()
            if fact_to_forget:
                knowledge_base.forget_fact(fact_to_forget, vector_database)
            else:
                print("🤖 Please provide the information you want me to forget after 'forget:'.")

        elif user_input.lower().startswith("koreksi:"):
            fact_to_correct = user_input[len("koreksi:"):].strip()
            if fact_to_correct:
                knowledge_base.handle_correction(fact_to_correct, vector_database)
            else:
                print("🤖 Please provide the correct information after 'koreksi:'.")
        
        elif user_input.lower().startswith("ringkas:"):
            text_to_summarize = user_input[len("ringkas:"):].strip()
            if text_to_summarize:
                llm_service.summarize_text(text_to_summarize)
            else:
                print("🤖 Please provide the text you want me to summarize after 'ringkas:'.")

        else:
            language = detect_language(user_input)
            question_embedding = llm_service.get_embedding(user_input, config.EMBEDDING_MODEL)
            
            relevant_context, sources, top_score = knowledge_base.find_relevant_chunks(question_embedding, vector_database)
            
            if top_score < config.CLARIFICATION_THRESHOLD:
                print("\n🤖 BramAI: I'm not quite sure what you mean. Could you try rephrasing the question?")
                continue

            ai_response = llm_service.generate_response(user_input, relevant_context, language, conversation_history, sources)

            if ai_response:
                print(f"\n🤖 BramAI: {ai_response}")
                if sources:
                    print(f"   💡 Sources: {', '.join(sources)}")

                conversation_history.append({"user": user_input, "ai": ai_response})

                if len(conversation_history) > config.CONVERSATION_HISTORY_LENGTH:
                    conversation_history.pop(0)

if __name__ == "__main__":
    main()
