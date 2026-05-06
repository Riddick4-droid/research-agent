from typing import TypedDict, List, Dict, Any
from memory.conversation_memory import summarize_history
from memory.research_memory import update_research_memory

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
    history: list
    conversation_summary:str
    research_memory:str


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

def memory_update_node(state: GraphState) -> GraphState:
    state["conversation_summary"] = summarize_history(state.get("history", []))
    state = update_research_memory(state)
    return state

def human_feedback_node(state: GraphState) -> GraphState:
    print("\nGenerated Subtopics:\n")
    for i, s in enumerate(state.get("subtopics", [])):
        print(f"{i+1}: {s}")
    choice = input("\nApprove subtopics? (y/n): ")
    if choice.lower() == 'y':
        return state
    else:
        new_query = input("Refine your query (comma separated): ")
        state["query"] = new_query
    return state

from langgraph.graph import StateGraph

def build_graph() -> StateGraph:
    graph = StateGraph(initial_state=GraphState(
        query="What are the latest advancements in quantum computing?",
        subtopics=[],
        current_index=0,
        current_subtopic="",
        retrieved_docs=[],
        results=[],
        synthesis="",
        reports="",
        history=[],  # Added default for history
        conversation_summary="",  # Added default for conversation_summary
        research_memory=""  # Added default for research_memory
    ))
    # Add nodes and edges to the graph
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("writer", writer_node)
    graph.add_node("next_subtopic", next_subtopic)
    graph.add_node("memory_update", memory_update_node)
    graph.add_node("human_feedback", human_feedback_node)

    #flow: planner -> research -> summarizer -> next_subtopic -> research (loop for each subtopic) -> synthesis -> writer
    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner", next_subtopic,
        {
            "research": "research",
            "synthesis": "synthesis"
        }

    )
    # Removed add_edge("planner", "human_feedback") to avoid flow conflicts
    graph.add_edge("research", "summarizer")
    graph.add_edge("summarizer", "memory_update")
    graph.add_conditional_edges(
        "summarizer", next_subtopic,
        {
            "research": "research",
            "synthesis": "synthesis"
        }
    )
    graph.add_edge("synthesis", "writer")
    graph.add_edge("human_feedback", "planner")

    graph.set_finish_point("writer")

    return graph.compile()