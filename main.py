from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.synthesis_agent import SynthesisAgent
from agents.writer_agent import WriterAgent
from utils.logger import get_logger

def main():
    logger = get_logger("Main")
    logger.info("Starting Research Agent...")

    planner = PlannerAgent()
    researcher = ResearchAgent()
    summarizer = SummarizerAgent()
    synthesis = SynthesisAgent()
    writer = WriterAgent()

    state = {"query": "What are the latest advancements in quantum computing?"}
    logger.info("Running PlannerAgent...")
    state = planner.run(state)
    logger.info("Running ResearchAgent...")
    for subtopic in state.get("subtopics", []):
        logger.info(f"Processing subtopic: {subtopic}")
        state["current_subtopic"] = subtopic

        state = researcher.run(state)
        state = summarizer.run(state)

        retrieved_docs = state.get("retrieved_docs", [])
        logger.info(f"Retrieved {len(retrieved_docs)} documents for subtopic '{subtopic}'.")
        logger.info(f"Summary for subtopic '{subtopic}': {state['results'][-1]['summary']}")  # Log the latest summary result
        for doc in retrieved_docs:
            logger.info(f"Document snippet: {doc[:200]}...")  # Log the first 200 characters of each retrieved document

    logger.info("Running SynthesisAgent...")
    state = synthesis.run(state)
    logger.info("Running WriterAgent...")
    state = writer.run(state)
    logger.info("Research Agent finished.")
if __name__ == "__main__":
    main()
