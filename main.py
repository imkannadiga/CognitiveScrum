import os
from crewai import Crew, Process
from src.agents import CognitiveScrumAgents
from src.tasks import CognitiveScrumTasks

dummy_resume_text = """
JANE DOE
Senior Full Stack Engineer
Summary: 8 years of experience in Python, React, and AWS. 
Specialized in backend optimization and microservices.
Experience:
- Lead Developer at TechCorp (2019-Present): Migrated monolith to microservices using FastAPI.
- Senior Dev at StartUpInc (2015-2019): Built real-time dashboard using React and Redux.
"""

dummy_backlog = [
    {
        "ticket_id": "T-101",
        "description": "Migrate user authentication service from legacy code to FastAPI microservice.",
        "complexity": "High",
        "required_skills": ["Python", "Microservices", "FastAPI"]
    },
    {
        "ticket_id": "T-102",
        "description": "Fix CSS alignment bug on the login page.",
        "complexity": "Low",
        "required_skills": ["CSS", "React"]
    }
]

def main():
    # --- 2. Initialize Agents & Tasks ---
    agents = CognitiveScrumAgents()
    tasks = CognitiveScrumTasks()

    # Instantiate Agents
    parser = agents.resume_parser_agent()
    planner = agents.sprint_planner_agent()
    critic = agents.critic_agent()

    # Instantiate Tasks
    # Task 1: Parse the raw resume text into structured JSON
    task_parse = tasks.parse_resumes_task(parser, dummy_resume_text)

    # Task 2: Plan the sprint (Assign tickets based on the parsed profile)
    task_plan = tasks.plan_sprint_task(planner, [dummy_resume_text], dummy_backlog)

    # Task 3: Critique the plan (Check for hallucinations or bias)
    task_critique = tasks.critique_plan_task(critic, task_plan)

    # --- 3. Instantiate the Crew ---
    # The 'Sequential' process ensures Task 1 -> Task 2 -> Task 3
    scrum_crew = Crew(
        agents=[parser, planner, critic],
        tasks=[task_parse, task_plan, task_critique],
        process=Process.sequential,
        verbose=True
    )

    # --- 4. Kickoff ---
    print("### Starting CognitiveScrum Run ###")
    result = scrum_crew.kickoff()
    
    print("\n\n########################")
    print("## FINAL CRITIC REPORT ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    main()