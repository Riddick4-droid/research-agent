from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class for tools."""
    @abstractmethod
    def run(self, query:str, **kwargs) -> str:
        """Execute the tool with the given arguments."""
        pass