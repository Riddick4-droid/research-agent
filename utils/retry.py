import time
from utils.logger import get_logger
logger = get_logger(__name__)

def retry_on_exception(func, max_retries=3, delay=1, backoff=2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff
            else:
                logger.error(f"Attempt {attempt + 1} failed with error: {e}. No more retries left.")
                raise