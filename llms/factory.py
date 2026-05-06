from llms.hf_llm import HuggingFaceLLM
from llms.openai_llm import OpenAILLM
from llms.anthropic_llm import AnthropicLLM

from config import PROVIDER, MODEL_NAME

def get_llm():
    """Factory function to get the appropriate LLM based on configuration."""
    if PROVIDER == "huggingface":
        return HuggingFaceLLM(model_name=MODEL_NAME)
    elif PROVIDER == "openai":
        return OpenAILLM(model_name=MODEL_NAME)
    elif PROVIDER == "anthropic":
        return AnthropicLLM(model_name=MODEL_NAME)
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")
    
