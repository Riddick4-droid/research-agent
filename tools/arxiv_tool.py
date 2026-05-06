import arxiv
from tools.base_tool import BaseTool
from utils.logger import get_logger
import os
from dotenv import load_dotenv

logger = get_logger(__name__)

#setup arxiv client
client = arxiv.Client()

class ArxivTool(BaseTool):
    """Tool for searching and retrieving information from arXiv."""
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ARXIV_API_KEY")
        logger.info("ArxivTool initialized.")

    def run(self, query: str, max_results: int = 5) -> str:
        """Search arXiv for papers matching the query and return a summary."""
        logger.debug(f"Running ArxivTool with query: {query}")
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = []
            for result in client.results(search):
                paper_info = f"Title: {result.title}\nAuthors: {', '.join(author.name for author in result.authors)}\nPublished: {result.published}\nSummary: {result.summary}\nURL: {result.entry_id}\n"
                results.append(paper_info)
            logger.debug(f"ArxivTool found {len(results)} results.")
            if not results:
                return "No papers found matching the query."
            else:
                return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error running ArxivTool: {e}")
            return "An error occurred while searching arXiv."