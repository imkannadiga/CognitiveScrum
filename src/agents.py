from crewai import Agent
from .config import llm_config
from src.tools.resume_parser import parse_pdf 

class CognitiveScrumAgents:
    def __init__(self):
        self.llm = llm_config['model']

    def resume_parser_agent(self):
        return Agent(
            role='Data Parser',
            goal='Extract structured skills and seniority data from resume text.',
            backstory=(
                "You are an expert HR data analyst. Your job is to convert unstructured "
                "PDF text into strict JSON formats. You do not invent skills; you only "
                "extract what is explicitly stated to mitigate 'Hallucination Risks'."
            ),
            #tools=[parse_pdf], 
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    def sprint_planner_agent(self):
        return Agent(
            role='Senior Scrum Master',
            goal='Create an optimal sprint schedule based on capacity and skill matching.',
            backstory=(
                "You are a veteran Scrum Master. You despise 'Subjective Bias' and "
                "assign tasks solely based on semantic skill matching and seniority. "
                "You calculate capacity granularly, ensuring no one is overloaded."
            ),
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    def critic_agent(self):
        return Agent(
            role='Guardrail Auditor',
            goal='Validate the sprint schedule for feasibility and logic.',
            backstory=(
                "You are a skeptic. You review the Sprint Planner's schedule to catch "
                "'Hallucinations' or impossible deadlines. You demand 'Reasoning Traces' "
                "for every assignment (e.g., 'Why was Junior Dev A assigned High Risk Task B?')."
            ),
            allow_delegation=True,
            verbose=True,
            llm=self.llm
        )