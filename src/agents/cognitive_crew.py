"""
CrewAI Agent Definitions for CognitiveScrum.
Includes the congnitive scrum crew (Staffing Expert, Scheduler, Critic).
"""
from crewai import Agent, Crew, Process, Task
from typing import List, Dict, Optional
from src.config import ModelConfig


class CognitiveScrumAgents:
    """CrewAI Agents for Sprint Planning."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = model_config.get_llm()
    
    def staffing_expert_agent(self):
        """Agent 1: Maps skills from resumes to backlog items."""
        return Agent(
            role='Staffing Expert',
            goal='Match team member skills from resumes to backlog items based on semantic similarity and expertise.',
            backstory=(
                "You are an expert HR and technical recruiter with 15 years of experience. "
                "You analyze resumes deeply, extracting not just explicit skills but also "
                "inferred capabilities from experience. You match these to backlog items with "
                "precision, considering both technical skills and seniority levels. "
                "You provide clear reasoning traces for each match."
            ),
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def scheduler_agent(self):
        """Agent 2: Assigns tasks based on capacity and seniority."""
        return Agent(
            role='Sprint Scheduler',
            goal='Create an optimal sprint schedule assigning tasks to team members based on capacity, skills, and seniority.',
            backstory=(
                "You are a veteran Scrum Master with 20 years of experience managing agile teams. "
                "You calculate capacity granularly, ensuring no one is overloaded. "
                "You respect seniority: Junior developers need 1.5x time for complex tasks. "
                "You balance workload evenly and consider dependencies. "
                "You always provide time estimates and risk assessments."
            ),
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def critic_agent(self):
        """Agent 3: Validates the plan for hallucinations and feasibility."""
        return Agent(
            role='Guardrail Auditor',
            goal='Validate the sprint schedule for feasibility, logic, and absence of hallucinations.',
            backstory=(
                "You are a skeptical, detail-oriented auditor. You review sprint plans with "
                "a critical eye, checking for: impossible deadlines, skill mismatches, "
                "capacity overloads, missing dependencies, and any 'hallucinated' assignments "
                "that don't match the actual team capabilities. "
                "You demand reasoning traces for every assignment and flag risks immediately."
            ),
            allow_delegation=True,
            verbose=True,
            llm=self.llm
        )
    
    def create_planning_crew(self, combined_context: str) -> tuple:
        """
        Create and configure the CrewAI crew for sprint planning.
        
        Args:
            combined_context: Combined context from ChromaDB (resumes, backlog, project context)
            
        Returns:
            Tuple of (Crew object, scheduler_task, critic_task) for accessing individual outputs
        """
        # Initialize agents
        staffing_expert = self.staffing_expert_agent()
        scheduler = self.scheduler_agent()
        critic = self.critic_agent()
        
        # Create tasks
        task1_staffing = Task(
            description=(
                f"Analyze the following project context and team resumes:\n\n{combined_context}\n\n"
                "Your task:\n"
                "1. Extract all team member skills and seniority levels from resumes\n"
                "2. Extract all backlog items with their required skills and complexity\n"
                "3. Match each backlog item to the best-suited team member(s) based on skills\n"
                "4. Provide reasoning traces for each match (e.g., 'John assigned to T-101 because he has 5 years FastAPI experience')\n\n"
                "Output format: A structured analysis mapping backlog items to potential assignees with reasoning."
            ),
            agent=staffing_expert,
            expected_output="A structured mapping of backlog items to team members with detailed reasoning traces."
        )
        
        task2_scheduling = Task(
            description=(
                "Using the staffing analysis from the previous task, create a detailed sprint schedule.\n\n"
                "Requirements:\n"
                "- Assign each backlog item to a specific team member\n"
                "- Estimate hours for each task (consider complexity and seniority)\n"
                "- Calculate total capacity per team member (assume 40 hours/week, adjust for availability)\n"
                "- Ensure no one is overloaded\n"
                "- Include risk assessment (Low/Medium/High) for each assignment\n\n"
                "CRITICAL: Output MUST be in a markdown table format with the following columns:\n"
                "| Task_ID | Assignee | Estimated_Hours | Risk_Level | Reasoning_Trace |\n"
                "|---------|----------|-----------------|------------|------------------|\n"
                "| T-101   | John Doe | 8               | Low        | ...              |\n\n"
                "This table format is essential for proper parsing and display."
            ),
            agent=scheduler,
            expected_output="A markdown table with columns: Task_ID, Assignee, Estimated_Hours, Risk_Level, Reasoning_Trace. Each row represents one task assignment."
        )
        
        task3_critique = Task(
            description=(
                "Review the sprint schedule from the previous task. Validate:\n"
                "1. Are all assignments feasible given team member skills?\n"
                "2. Are time estimates realistic?\n"
                "3. Is anyone overloaded (exceeding capacity)?\n"
                "4. Are there any hallucinations (assignments that don't match actual skills)?\n"
                "5. Are dependencies considered?\n\n"
                "Provide a final validation report with any flagged issues or approval."
            ),
            agent=critic,
            expected_output="A validation report with approval status and any flagged risks or issues."
        )
        
        # Create crew
        crew = Crew(
            agents=[staffing_expert, scheduler, critic],
            tasks=[task1_staffing, task2_scheduling, task3_critique],
            process=Process.sequential,
            verbose=True
        )
        
        # Return crew and task references for accessing individual outputs
        return crew, task2_scheduling, task3_critique
