import anthropic
from llms.base import BaseLLM
from utils.logger import get_logger
import os
logger = get_logger(__name__)

class AnthropicLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info(f"Anthropic LLM initialized with model: {model_name}")
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate a response based on the given prompt."""
        logger.debug(f"Generating response for prompt: {prompt}")
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated response: {generated_text}")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
