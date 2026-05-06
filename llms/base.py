from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for language models."""
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response based on the given prompt."""
        pass