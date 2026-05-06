from agents.base_agent import BaseAgent
from tools.arxiv_tool import ArxivTool
from tools.text_processing import combine_papers, split_text, remove_duplicates
from tools.vector_store import create_vector_store, get_embedding_model, retrieve
from utils.logger import get_logger

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ResearchAgent")
        self.tool = ArxivTool()
        self.logger = get_logger(self.name)

    def run(self, state: dict) -> dict:
        subtopic = state.get("subtopics", []) 
        self.logger.info(f"Processing subtopic: {subtopic}")
        if not subtopic:
            self.logger.warning("No subtopics found in state.")
            return state
        papers = self.tool.run(subtopic, max_results=5)
        if papers:
            self.logger.info(f"Found papers for subtopic '{subtopic}'.")
        else:
            self.logger.warning(f"No papers found for subtopic '{subtopic}'.")
            state["retrieved_docs"] = []
            return state
        combined = combine_papers(papers)
        chunks = split_text(combined)
        unique_chunks = remove_duplicates(chunks)
        embedding_model = get_embedding_model()
        vector_store = create_vector_store(unique_chunks, embedding_model)
        docs = retrieve(vector_store,subtopic,k=5)
        state["retrieved_docs"] = docs
        return state
