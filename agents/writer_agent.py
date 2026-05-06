from agents.base_agent import BaseAgent
from utils.logger import get_logger
from llms.factory import get_llm
from utils.cleaner import clean_output

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="WriterAgent")
        self.llm = get_llm()
        self.logger = get_logger(self.name)
    
    def build_prompt(self,synthesis,query):
        return f"""
You are writing a high-quality academic research document based on the following synthesis of research findings related to the query: "{query}".

Topic:
{query}

Synthesis:
{synthesis}

Task:
Write a structured report with the following sections:
1. Abstract: A concise summary of the key findings and insights.
2. Introduction: An introduction to the topic and its significance.
3. Methodological overview: Discuss key techniques and approaches identified in the synthesis.
4. Applications: Real-world use cases
5. Limitations and Challenges: Critical analysis of limitations and challenges in the current research.
6. Future Directions: Suggestions for future research based on the synthesis.
7. Conclusion: A summary of the main insights and implications.
Rules:
- Be clear, concise, and well-structured.
- Use formal academic language.
- Do NOT repeat the synthesis verbatim, but use it to inform your writing.
- Focus on providing a comprehensive and insightful report based on the synthesis.
"""
    def run(self,state:dict)->dict:
        synthesis = state.get("synthesis","")
        query = state.get("query","")
        self.logger.info(f"Running WriterAgent for query: {query}")
        if not synthesis:
            self.logger.warning("No synthesis available for writing.")
            state["reports"] = "No information available to write a report."  # Changed from "report" to "reports" to match GraphState
            return state
        self.logger.info("Building prompt for writing...")
        prompt = self.build_prompt(synthesis,query)
        self.logger.info("Generating report using LLM...")
        raw_report = self.llm.generate(prompt)
        self.logger.info("Report generated, cleaning output...")
        report = clean_output(raw_report)
        state["reports"] = report  # Changed from "report" to "reports" to match GraphState
        return state