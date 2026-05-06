from abc import ABC, abstractmethod
from utils.loggers import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Abstract base class for agents."""
    def __init__(self,name:str):
        self.name=name
        self.logger=get_logger(name)
    @abstractmethod
    def act(self, state: dict) -> dict:
        """Perform an action based on the given observation."""
        pass