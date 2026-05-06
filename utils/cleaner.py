#cleaner utility to clean the output from LLMs, removing unnecessary whitespace, newlines, and ensuring a consistent format.
import re
def clean_output(text):
    """Clean the output text by removing extra whitespace and newlines."""
    # Remove leading/trailing whitespace
    cleaned = text.strip()
    # Replace multiple newlines with a single newline
    cleaned = re.sub(r'\n+', '\n', cleaned)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned