from utils.retry import retry_on_exception
from utils.logger import get_logger

logger = get_logger(__name__)

response = retry_on_exception(lambda: "This is a test response from the retry mechanism.")