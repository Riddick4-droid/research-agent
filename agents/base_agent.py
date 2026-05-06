from abc import ABC, abstractmethod
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Abstract base class for agents."""
    def __init__(self,name:str):
        self.name=name
        self.logger=get_logger(name)
    @abstractmethod
    def run(self, state: dict) -> dict:  # Changed from 'act' to 'run' to match subclass implementations
        """Perform an action based on the given observation."""
        pass