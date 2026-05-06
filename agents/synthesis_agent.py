from agents.base_agent import BaseAgent
from llms.factory import get_llm
from utils.logger import get_logger
from utils.cleaner import clean_output

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SynthesisAgent")
        self.llm = get_llm()
        self.logger = get_logger(self.name)
    def build_prompt(self,summaries,query):
        combined = ""

        for item in summaries:
            combined += f"\n\nSubtopic: {item['subtopic']}\nSummary: {item['summary']}\n"

        return f"""  # Moved return outside the loop to process all summaries
You are performing a deep research analysis compiled from multiple sources. Your task is to synthesize the following summaries related to the query: "{query}".

Content:
{combined}

Task:
Produce an analytical synthesis of the information for the {query}.

Output:

Core Themes:
- Identify the most important recurring ideas

Comparative Analysis:
- Compare techniques and approaches

Key Techniques Ranked:
- Rank the most effective techniques based on the summaries (most -> least effective)

Applications and Implications:
- Discuss potential applications and implications of the findings

Critical limitations:
- Identify critical limitations and gaps in the current research

Insights and Future Directions:
- Provide insights and suggest future research directions based on the synthesis

Rules:
- Do NOT repeat the content verbatim
- Merge similar ideas and avoid redundancy
- Be analytical and critical, not just descriptive
"""
    def run(self,state:dict)->dict:
        summaries = state.get("results",[])
        query = state.get("query","")
        self.logger.info(f"Running synthesis for query: {query}")
        if not summaries:
            self.logger.warning("No summaries available for synthesis.")
            state["synthesis"] = "No information available to synthesize."
            return state
        self.logger.info("Running synthesis...")
        prompt = self.build_prompt(summaries,query)
        self.logger.info("Generating synthesis using LLM...")
        raw_synthesis = self.llm.generate(prompt)
        self.logger.info("Synthesis generated, cleaning output...")
        synthesis = clean_output(raw_synthesis)
        state["synthesis"] = synthesis
        return state

