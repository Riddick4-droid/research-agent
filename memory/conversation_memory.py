from llms.factory import get_llm
from utils.logger import get_logger

llm = get_llm()
logger = get_logger("ConversationMemory")

def summarize_history(history,max_messages:int=5)->str:
    """Summarize the conversation history to keep it concise."""
    if not history:
        return "No conversation history."
    
    # Keep only the last max_messages interactions
    recent_history = history[-max_messages:]
    
    # Create a summary string
    text = ""
    for entry in recent_history:
        text += f"{entry['role']}: {entry['content']}\n"
    
    logger.info("Summarizing conversation history...")
    prompt = f"""Summarize the following conversation 
    history while maintaining the key points:\n{text}\n\nSummary:
    - Keep it concise (1-2 sentences)
    - Focus on the main topics discussed
    - Do NOT include irrelevant details
    """
    summarized_history = llm.generate(prompt, max_tokens=100, temperature=0.5)
    
    return summarized_history.strip()