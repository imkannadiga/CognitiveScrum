from crewai import Agent, Crew, Process, Task
from typing import List, Dict, Optional
from src.config import ModelConfig

class InterviewerAgent:
    """
    Lightweight interviewer agent that asks questions and determines context sufficiency.
    This is not a CrewAI agent, but a simple LLM-based function.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = model_config.get_llm()
    
    def generate_question(self, chat_history: List[Dict], existing_context: str = "") -> Dict:
        """
        Generate the next question based on chat history.
        Returns a dict with 'question' and 'sufficiency_score' (0-100).
        
        Args:
            chat_history: List of dicts with 'role' and 'content'
            existing_context: Existing project context from ChromaDB
            
        Returns:
            Dict with 'question', 'sufficiency_score', and 'ready_to_plan' boolean
        """
        # Build conversation context
        conversation_text = ""
        for msg in chat_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            conversation_text += f"{role.upper()}: {content}\n"
        
        prompt = f"""You are an expert Scrum Master conducting a discovery interview to gather PROJECT REQUIREMENTS ONLY.

                        IMPORTANT: Do NOT ask about the team members, their skills, or team composition. 
                        Team information will be provided separately through resume uploads.
                        Focus ONLY on the PROJECT itself.

                        EXISTING CONTEXT:
                        {existing_context}

                        CONVERSATION SO FAR:
                        {conversation_text}

                        Your task:
                        1. Analyze what PROJECT information is still missing (deadlines, tech stack, project priorities, scope, constraints, etc.)
                        2. Generate ONE specific, focused question about the PROJECT to fill the most critical gap
                        3. Assess context completeness (0-100%): How much do we know about:
                        - Project deadlines and timeline
                        - Technical requirements and technology stack
                        - Project priorities and scope
                        - Resource limitations and constraints
                        - Business requirements and objectives

                        DO NOT ask about:
                        - Team members or their skills
                        - Team composition or availability
                        - Individual capabilities

                        Respond in this EXACT format:
                        QUESTION: [your question here]
                        SUFFICIENCY_SCORE: [0-100]
                        READY_TO_PLAN: [true/false] (true if score >= 80)

                        If READY_TO_PLAN is true, the question can be "All context gathered. Ready to generate sprint plan."
                    """
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            question = ""
            sufficiency_score = 0
            ready_to_plan = False
            
            for line in response_text.split('\n'):
                if line.startswith('QUESTION:'):
                    question = line.replace('QUESTION:', '').strip()
                elif line.startswith('SUFFICIENCY_SCORE:'):
                    try:
                        sufficiency_score = int(line.replace('SUFFICIENCY_SCORE:', '').strip())
                    except:
                        sufficiency_score = 0
                elif line.startswith('READY_TO_PLAN:'):
                    ready_str = line.replace('READY_TO_PLAN:', '').strip().lower()
                    ready_to_plan = ready_str == 'true' or sufficiency_score >= 80
            
            if not question:
                question = "Could you tell me more about the project timeline and deadlines?"
            
            return {
                "question": question,
                "sufficiency_score": min(100, max(0, sufficiency_score)),
                "ready_to_plan": ready_to_plan
            }
        except Exception as e:
            # Fallback question
            return {
                "question": "What are the key deadlines and priorities for this sprint?",
                "sufficiency_score": 30,
                "ready_to_plan": False
            }