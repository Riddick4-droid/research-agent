from graph.research_graph import build_graph
from utils.logger import get_logger

def main():
    logger = get_logger("Main")
    logger.info("Starting Research Agent...")

    graph = build_graph()
    state = {"query": "Vision transformers in healthcare?",
             "current_index": 0,
                "results": []}
    
    final_state = graph.invoke(state)
    logger.info("Research Agent finished.")

    logger.info("Final synthesis:")
    logger.info(final_state.get("synthesis", "No synthesis generated."))

if __name__ == "__main__":
    main()