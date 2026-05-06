from agents.base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="PlannerAgent")
    
    def run(self, state:dict)->dict:
        query = state.get("query","")
        self.logger.info(f"Received query: {query}")
        self.logger.info('Generating research plan...')

        subtopics = [
            f"Overview of {query}",
            f"Key concepts related to {query}",
            f"Applications of {query}",
            f"Recent advancements in {query}",
            f"Challenges associated with {query}",
            f"limitations of current research on {query}",
            f"Future directions for research on {query}"
        ]
        state["subtopics"] = subtopics
        self.logger.info(f"Generated subtopics: {subtopics}")
        state["results"] = []

        return state