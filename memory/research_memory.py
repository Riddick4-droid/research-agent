from utils.logger import get_logger

logger = get_logger(__name__)

def update_research_memory(state: dict):
    summaries = state.get("results", [])
    if not summaries:
        logger.warning("No summaries found in state to update research memory.")
        return state
    insights = []
    for item in summaries:
        subtopic = item.get("subtopic", "Unknown Subtopic")
        summary = item.get("summary", "")
        insights.append(f"Subtopic: {subtopic}\nSummary: {summary}")
    state["research_memory"] = "\n\n".join(insights)
    logger.info("Research memory updated with latest summaries.")
    return state

