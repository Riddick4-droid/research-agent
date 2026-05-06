from transformers import pipeline
from llms.base import BaseLLM
from utils.retry import retry_on_exception
from utils.logger import get_logger

#quantization
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16"
}

logger = get_logger(__name__)

class HuggingFaceLLM(BaseLLM):
    """Hugging Face language model implementation."""
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.generator = pipeline('text-generation', 
                                  model=model_name,
                                  device_map='auto',
                                  do_sample=False,
                                  quantization_config=quantization_config)
        logger.info(f"Hugging Face LLM initialized with model: {model_name}")

    def generate(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1) -> str:
        """Generate a response based on the given prompt."""
        logger.debug(f"Generating response for prompt: {prompt}")
        try:
            responses = retry_on_exception(lambda: self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences))
            generated_texts = [response['generated_text'] for response in responses]
            logger.debug(f"Generated responses: {generated_texts}")
            return generated_texts[0] if generated_texts else ""
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""