from fastapi import FastAPI
from graph.research_graph import build_graph
from utils.logger import get_logger

app =FastAPI()
graph = build_graph()

@app.post("/research")
def run_research(query: str):
    state = {
        "query":query,
        "current_index":0,
        "results":[],
        "history":[]
    }
    result = graph.invoke(state)
    return {'report':result['reports']}