from crewai import Task

class CognitiveScrumTasks:
    def parse_resumes_task(self, agent, resume_text):
        return Task(
            description=(
                f"Analyze the following resume text: '{resume_text}'. "
                "Extract a JSON object containing: Name, Years_Experience, "
                "Skills_List, and Seniority_Level."
            ),
            agent=agent,
            expected_output="A structured JSON object representing the candidate's profile."
        )

    def plan_sprint_task(self, agent, employees, backlog):
        return Task(
            description=(
                f"Using the following employee profiles: {employees} "
                f"and this sprint backlog: {backlog}, assign tasks to employees. "
                "Match skills in the backlog tickets to skills in employee profiles. "
                "Respect seniority: Juniors need 1.5x time for complex tasks."
            ),
            agent=agent,
            expected_output="A Sprint Schedule Matrix mapping tasks to employees with time estimates."
        )

    def critique_plan_task(self, agent, sprint_plan):
        return Task(
            description=(
                f"Review this sprint plan: {sprint_plan}. "
                "Identify any risks, unassigned high-priority tasks, or capacity overloads. "
                "Provide a 'Reasoning Trace' for why the plan is or isn't viable."
            ),
            agent=agent,
            expected_output="A text-based Risk Analysis and Final Approval status."
        )