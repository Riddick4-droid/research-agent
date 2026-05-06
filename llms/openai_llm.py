from openai import OpenAI
from llms.base import BaseLLM
from utils.logger import get_logger
from utils.retry import retry_on_exception
import os

logger = get_logger(__name__)

class OpenAILLM(BaseLLM):
    """OpenAI language model implementation."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI LLM initialized.")

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate a response based on the given prompt."""
        logger.debug(f"Generating response for prompt: {prompt}")
        try:
            response = retry_on_exception(lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            ),max_retries=2,delay=2)
            generated_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated response: {generated_text}")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""