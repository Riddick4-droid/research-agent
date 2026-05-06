from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """TypedDict to represent the state of the research graph."""
    query: str
    subtopics: List[str]
    current_index: int
    current_subtopic: str
    retrieved_docs: List[str]
    results: List[Dict]
    synthesis: str
    reports: str


from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.synthesis_agent import SynthesisAgent
from agents.writer_agent import WriterAgent
from utils.logger import get_logger

planner = PlannerAgent()
researcher = ResearchAgent()
summarizer = SummarizerAgent()
synthesis = SynthesisAgent()
writer = WriterAgent()


def planner_node(state: GraphState) -> GraphState:
    return planner.run(state)

def research_node(state: GraphState) -> GraphState:
    return researcher.run(state)

def summarizer_node(state: GraphState) -> GraphState:
    return summarizer.run(state)

def synthesis_node(state: GraphState) -> GraphState:
    return synthesis.run(state)

def writer_node(state: GraphState) -> GraphState:
    return writer.run(state)

def next_subtopic(state: GraphState) -> GraphState:
    idx = state.get("current_index", 0) 

    if idx >= len(state.get("subtopics", [])):
        return "synthesis"
    state["current_subtopic"] = state["subtopics"][idx]
    state["current_index"] = idx + 1
    return "research"

from langraph.graph import StateGraph

def build_graph() -> StateGraph:
    graph = StateGraph(initial_state=GraphState(
        query="What are the latest advancements in quantum computing?",
        subtopics=[],
        current_index=0,
        current_subtopic="",
        retrieved_docs=[],
        results=[],
        synthesis="",
        reports=""
    ))
    # Add nodes and edges to the graph
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("writer", writer_node)
    graph.add_node("next_subtopic", next_subtopic)

    #flow: planner -> research -> summarizer -> next_subtopic -> research (loop for each subtopic) -> synthesis -> writer
    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner", next_subtopic,
        {
            "research": "research",
            "synthesis": "synthesis"
        }

    )
    graph.add_edge("research", "summarizer")
    graph.add_conditional_edges(
        "summarizer", next_subtopic,
        {
            "research": "research",
            "synthesis": "synthesis"
        }
    )
    graph.add_edge("synthesis", "writer")

    graph.set_finish_point("writer")

    return graph.compile()