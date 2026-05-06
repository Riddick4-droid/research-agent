from agents.base_agent import BaseAgent
from llms.factory import get_llm
from utils.cleaner import clean_output
from utils.logger import get_logger

class SummarizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SummarizerAgent")
        self.llm = get_llm()
        self.logger = get_logger(self.name)
    
    def build_prompt(self,docs,subtopic):
        context = "\n\n".join(docs)
        return f"""
Summarize the following research content.
Context:
{context}
Task:
Extract ONLY relevant information related to the subtopic: "{subtopic}".

Output format:

Key insights:
- Bullet points only

Techniques:
- Bullet points only

Applications:
- Bullet points only

Challenges:
- Bullet points only

Limitations:
- Bullet points only

Rules:
- Do NOT  repeat the context
- Do NOT  include instructions
- Be concise and focus on the subtopic
"""
    def run(self, state:dict)->dict:
        docs = state.get("retrieved_docs",[])
        subtopic = state['current_subtopic']
        self.logger.info(f"Summarizing information for subtopic: {subtopic}")
        if not docs:
            self.logger.warning("No documents to summarize.")
            state["summary"] = "No relevant information found."
            return state
        self.logger.info(f"Building prompt for summarization for subtopics: {subtopic}")
        prompt = self.build_prompt(docs,subtopic)
        self.logger.info("Generating summary using LLM...")
        raw_summary = self.llm.generate(prompt)
        self.logger.info("Summary generated, cleaning output...")
        summary = clean_output(raw_summary)

        if "results" not in state:
            state["results"] = []
        state["results"].append({
            "subtopic": subtopic,
            "summary": summary
        })
        self.logger.info(f"Summary for subtopic '{subtopic}' added to state.")
        return state
    