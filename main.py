"""
CognitiveScrum (OptiSprint) - Autonomous Scrum Master Application
Streamlit-based interface for sprint planning with AI agents.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict
import os
from datetime import datetime
import warnings
import logging

# Suppress CrewAI telemetry warnings (signal handlers don't work in Streamlit's thread context)
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "1"

# Suppress various warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*signal only works in main thread.*")
warnings.filterwarnings("ignore", message=".*Cannot register.*handler.*")

# Suppress CrewAI telemetry logging
logging.getLogger("crewai.telemetry").setLevel(logging.ERROR)

from src.config import ModelConfig
from src.db_handler import DBHandler
from src.agents.interview_agent import InterviewerAgent
from src.agents.cognitive_crew import CognitiveScrumAgents
from src.utils import parse_pdf, parse_csv_backlog, parse_json_backlog, extract_candidate_name, parse_sprint_plan_output


# Page configuration
st.set_page_config(
    page_title="CognitiveScrum - Autonomous Scrum Master",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if "model_config" not in st.session_state:
    st.session_state.model_config = ModelConfig()

if "db_handler" not in st.session_state:
    st.session_state.db_handler = DBHandler()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sufficiency_score" not in st.session_state:
    st.session_state.sufficiency_score = 0

if "ready_to_plan" not in st.session_state:
    st.session_state.ready_to_plan = False

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "pending_question_generation" not in st.session_state:
    st.session_state.pending_question_generation = False

if "sprint_plan" not in st.session_state:
    st.session_state.sprint_plan = None

if "sprint_plan_table" not in st.session_state:
    st.session_state.sprint_plan_table = None

if "sprint_plan_full_text" not in st.session_state:
    st.session_state.sprint_plan_full_text = None

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama3"

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "base_url" not in st.session_state:
    st.session_state.base_url = "http://localhost:11434"


def initialize_interviewer():
    """Initialize the interviewer agent with current model config."""
    try:
        st.session_state.model_config.update_from_session_state(st.session_state)
        return InterviewerAgent(st.session_state.model_config)
    except Exception as e:
        st.error(f"Failed to initialize interviewer: {str(e)}")
        return None


def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # LLM Settings
        st.header("LLM Settings")
        model_name = st.text_input(
            "Model Name",
            value=st.session_state.model_name,
            help="e.g., llama3, gpt-4, claude-3-opus"
        )
        api_key = st.text_input(
            "API Key (Optional)",
            value=st.session_state.api_key,
            type="password",
            help="Required for cloud models (OpenAI, Anthropic)"
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            help="e.g., http://localhost:11434 for Ollama"
        )
        
        # Update session state
        st.session_state.model_name = model_name
        st.session_state.api_key = api_key
        st.session_state.base_url = base_url
        
        # Test connection button
        if st.button("🔌 Test Connection"):
            with st.spinner("Testing connection..."):
                try:
                    st.session_state.model_config.update_from_session_state(st.session_state)
                    success, message = st.session_state.model_config.test_connection()
                    if success:
                        st.success(f"✅ Connection successful!\n{message}")
                    else:
                        st.error(f"❌ Connection failed: {message}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Database Management
        st.header("Database")
        if st.button("🗑️ Clear Project Memory", type="secondary"):
            if st.session_state.db_handler:
                st.session_state.db_handler.reset_db()
                st.session_state.chat_history = []
                st.session_state.sufficiency_score = 0
                st.session_state.ready_to_plan = False
                st.session_state.current_question = ""
                st.session_state.sprint_plan = None
                st.session_state.sprint_plan_table = None
                st.session_state.sprint_plan_full_text = None
                st.success("Project memory cleared!")
                st.rerun()
        
        # Status indicators
        st.divider()
        st.header("Status")
        st.metric("Context Completeness", f"{st.session_state.sufficiency_score}%")
        
        # Count uploaded items
        try:
            resumes = st.session_state.db_handler.get_all_resumes()
            backlog_items = st.session_state.db_handler.get_all_backlog()
            st.info(f"📄 Resumes: {len(resumes)}\n📋 Backlog Items: {len(backlog_items)}")
        except:
            pass
    
    # Main content area
    st.title("🚀 CognitiveScrum - Autonomous Scrum Master")
    st.markdown("**An AI-powered Scrum Master that interviews you and generates optimized sprint plans.**")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "💬 Context Interview", "📅 Sprint Plan"])
    
    # Tab 1: Knowledge Base
    with tab1:
        st.header("Upload Project Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Upload Resumes (PDF)")
            resume_files = st.file_uploader(
                "Select PDF resume files",
                type=["pdf"],
                accept_multiple_files=True,
                key="resume_uploader"
            )
            
            if st.button("Process Resumes", key="process_resumes"):
                if resume_files:
                    with st.spinner("Processing resumes..."):
                        for resume_file in resume_files:
                            try:
                                resume_text = parse_pdf(resume_file)
                                candidate_name = extract_candidate_name(resume_text)
                                
                                metadata = {
                                    "name": candidate_name,
                                    "upload_date": datetime.now().isoformat(),
                                    "filename": resume_file.name
                                }
                                
                                candidate_id = st.session_state.db_handler.add_resume(
                                    resume_text,
                                    metadata
                                )
                                st.success(f"✅ Processed resume: {candidate_name}")
                            except Exception as e:
                                st.error(f"❌ Error processing {resume_file.name}: {str(e)}")
                else:
                    st.warning("Please upload at least one PDF file.")
        
        with col2:
            st.subheader("📋 Upload Backlog (CSV/JSON)")
            backlog_files = st.file_uploader(
                "Select backlog files (supports multiple files)",
                type=["csv", "json"],
                accept_multiple_files=True,
                key="backlog_uploader"
            )
            
            if st.button("Process Backlog", key="process_backlog"):
                if backlog_files:
                    with st.spinner("Processing backlog files..."):
                        try:
                            total_items = 0
                            for backlog_file in backlog_files:
                                if backlog_file.name.endswith('.csv'):
                                    backlog_items = parse_csv_backlog(backlog_file)
                                else:
                                    backlog_items = parse_json_backlog(backlog_file)
                                
                                for item in backlog_items:
                                    metadata = {
                                        "ticket_id": item.get("ticket_id", ""),
                                        "complexity": item.get("complexity", "Medium"),
                                        "required_skills": item.get("required_skills", ""),
                                        "upload_date": datetime.now().isoformat(),
                                        "source_file": backlog_file.name
                                    }
                                    
                                    item_id = st.session_state.db_handler.add_backlog_item(
                                        item.get("description", ""),
                                        metadata,
                                        item.get("ticket_id")
                                    )
                                    total_items += 1
                            
                            st.success(f"✅ Processed {total_items} backlog items from {len(backlog_files)} file(s)")
                        except Exception as e:
                            st.error(f"❌ Error processing backlog: {str(e)}")
                else:
                    st.warning("Please upload at least one CSV or JSON file.")
        
        # Display uploaded data
        st.divider()
        st.subheader("📊 Uploaded Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Candidates:**")
            try:
                resumes = st.session_state.db_handler.get_all_resumes()
                if resumes:
                    resume_data = []
                    for resume in resumes:
                        metadata = resume.get("metadata", {})
                        resume_data.append({
                            "Name": metadata.get("name", "Unknown"),
                            "Filename": metadata.get("filename", "N/A")
                        })
                    st.dataframe(pd.DataFrame(resume_data), width='stretch')
                else:
                    st.info("No resumes uploaded yet.")
            except Exception as e:
                st.error(f"Error loading resumes: {str(e)}")
        
        with col2:
            st.write("**Backlog Items:**")
            try:
                backlog_items = st.session_state.db_handler.get_all_backlog()
                if backlog_items:
                    backlog_data = []
                    for item in backlog_items:
                        metadata = item.get("metadata", {})
                        backlog_data.append({
                            "Ticket ID": metadata.get("ticket_id", "N/A"),
                            "Complexity": metadata.get("complexity", "N/A")
                        })
                    st.dataframe(pd.DataFrame(backlog_data), width='stretch')
                else:
                    st.info("No backlog items uploaded yet.")
            except Exception as e:
                st.error(f"Error loading backlog: {str(e)}")
    
    # Tab 2: Context Interview
    with tab2:
        st.header("💬 Project Discovery Interview")
        st.markdown("The AI will ask you questions to gather project context. Answer each question to build a complete picture.")
        
        # Initialize interviewer
        interviewer = initialize_interviewer()
        
        if not interviewer:
            st.error("Please configure LLM settings in the sidebar and test the connection.")
        else:
            # Display chat history with minimized view
            chat_container = st.container()
            
            with chat_container:
                # Show chat history in minimized expanders
                for idx, message in enumerate(st.session_state.chat_history):
                    role_icon = "👤" if message["role"] == "user" else "🤖"
                    role_label = "You" if message["role"] == "user" else "AI Assistant"
                    
                    with st.expander(f"{role_icon} {role_label}", expanded=False):
                        st.write(message["content"])
            
            # Context completeness indicator
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(st.session_state.sufficiency_score / 100)
            with col2:
                st.metric("Completeness", f"{st.session_state.sufficiency_score}%")
            
            # Generate initial question if chat is empty
            if not st.session_state.chat_history and not st.session_state.current_question:
                if st.button("🎯 Start Interview"):
                    with st.spinner("Generating first question..."):
                        try:
                            existing_context = st.session_state.db_handler.get_combined_context()
                            result = interviewer.generate_question([], existing_context)
                            question = result["question"]
                            st.session_state.sufficiency_score = result["sufficiency_score"]
                            st.session_state.ready_to_plan = result["ready_to_plan"]
                            
                            # Add to chat history (not as current_question to avoid duplication)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": question
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating question: {str(e)}")
            
            # File upload for chat
            uploaded_files = st.file_uploader(
                "📎 Upload documents (PDF, TXT, DOCX) to provide context",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True,
                key="chat_file_uploader"
            )
            
            # Process uploaded files if any
            uploaded_content = ""
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        if file.name.endswith('.pdf'):
                            content = parse_pdf(file)
                            uploaded_content += f"\n\n[Document: {file.name}]\n{content}"
                        elif file.name.endswith('.txt'):
                            content = str(file.read(), "utf-8")
                            uploaded_content += f"\n\n[Document: {file.name}]\n{content}"
                        elif file.name.endswith('.docx'):
                            # For DOCX, we'd need python-docx library
                            st.info(f"DOCX support coming soon. Please convert {file.name} to PDF or TXT.")
                    except Exception as e:
                        st.warning(f"Could not read {file.name}: {str(e)}")
            
            # Chat input
            user_input = st.chat_input("Type your answer here...")
            
            if user_input:
                # Combine user input with uploaded content
                full_user_input = user_input
                if uploaded_content:
                    full_user_input = f"{user_input}\n\n{uploaded_content}"
                
                # Add user message to chat immediately
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": full_user_input
                })
                
                # Save to ChromaDB
                try:
                    st.session_state.db_handler.add_context(
                        full_user_input,
                        {"timestamp": datetime.now().isoformat(), "question": st.session_state.current_question}
                    )
                except Exception as e:
                    st.warning(f"Could not save context: {str(e)}")
                
                # Set flag to generate next question
                st.session_state.pending_question_generation = True
                
                # Rerun immediately to show user message
                st.rerun()
            
            # Generate next question after user input is displayed
            if st.session_state.pending_question_generation:
                st.session_state.pending_question_generation = False
                with st.spinner("Analyzing your answer and generating next question..."):
                    try:
                        existing_context = st.session_state.db_handler.get_combined_context()
                        result = interviewer.generate_question(
                            st.session_state.chat_history,
                            existing_context
                        )
                        
                        next_question = result["question"]
                        st.session_state.sufficiency_score = result["sufficiency_score"]
                        st.session_state.ready_to_plan = result["ready_to_plan"]
                        
                        # Add AI question to chat (only once, not duplicated)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": next_question
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating question: {str(e)}")
            
            # Ready to plan indicator
            if st.session_state.ready_to_plan:
                st.success("✅ **Context is sufficient!** You can now generate the sprint plan in the 'Sprint Plan' tab.")
    
    # Tab 3: Sprint Plan
    with tab3:
        st.header("📅 Sprint Plan Generation")
        
        if not st.session_state.ready_to_plan:
            st.warning("⚠️ Please complete the Context Interview first. Context completeness must be at least 80%.")
            st.info("Go to the 'Context Interview' tab to answer questions and gather project context.")
        else:
            # Generate sprint plan button
            if st.button("🚀 Generate Sprint Plan", type="primary", width='stretch'):
                with st.spinner("🤖 AI agents are working... This may take a few minutes."):
                    try:
                        # Get combined context
                        combined_context = st.session_state.db_handler.get_combined_context()
                        
                        # Initialize agents
                        st.session_state.model_config.update_from_session_state(st.session_state)
                        scrum_agents = CognitiveScrumAgents(st.session_state.model_config)
                        
                        # Create and run crew
                        crew, scheduler_task, critic_task = scrum_agents.create_planning_crew(combined_context)
                        result = crew.kickoff()
                        
                        # Extract individual task outputs
                        # CrewAI stores task outputs in task.output after execution
                        try:
                            scheduler_output = str(scheduler_task.output) if hasattr(scheduler_task, 'output') and scheduler_task.output else None
                            critic_output = str(critic_task.output) if hasattr(critic_task, 'output') and critic_task.output else None
                            
                            # Fallback: if outputs not available, try to extract from crew tasks
                            if not scheduler_output or not critic_output:
                                if hasattr(crew, 'tasks') and len(crew.tasks) >= 3:
                                    scheduler_output = str(crew.tasks[1].output) if hasattr(crew.tasks[1], 'output') and crew.tasks[1].output else str(result)
                                    critic_output = str(crew.tasks[2].output) if hasattr(crew.tasks[2], 'output') and crew.tasks[2].output else str(result)
                                else:
                                    # Last resort: use full result
                                    scheduler_output = str(result)
                                    critic_output = str(result)
                        except Exception as e:
                            # Fallback to full result if extraction fails
                            st.warning(f"Could not extract individual task outputs: {str(e)}. Using full result.")
                            scheduler_output = str(result)
                            critic_output = str(result)
                        
                        # Parse scheduler output for table (this contains the task assignments)
                        task_df, _ = parse_sprint_plan_output(scheduler_output)
                        
                        # Store results
                        st.session_state.sprint_plan = critic_output  # Full report from critic
                        st.session_state.sprint_plan_table = task_df  # Table from scheduler
                        st.session_state.sprint_plan_full_text = critic_output  # Critic's validation report
                        
                        st.success("✅ Sprint plan generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating sprint plan: {str(e)}")
                        st.exception(e)
            
            # Display sprint plan if exists
            if st.session_state.sprint_plan:
                st.divider()
                st.subheader("📊 Generated Sprint Plan")
                
                # Display task assignment table if available
                if st.session_state.sprint_plan_table is not None and not st.session_state.sprint_plan_table.empty:
                    st.markdown("### 📋 Task Assignments")
                    st.dataframe(
                        st.session_state.sprint_plan_table,
                        width='stretch',
                        hide_index=True
                    )
                    st.markdown("---")
                
                # Display full text response from Guardrail Auditor
                st.markdown("### 📝 Full Planning Report & Validation")
                full_text = st.session_state.sprint_plan_full_text or st.session_state.sprint_plan
                st.text_area(
                    "Complete Analysis Report",
                    full_text,
                    height=400,
                    help="This includes the Staffing Expert analysis, Scheduler's task assignments, and Guardrail Auditor's validation report."
                )
                
                # Correction input
                st.divider()
                st.subheader("🔄 Plan Corrections")
                st.markdown("If you need to make corrections (e.g., 'Sarah is on vacation'), enter them below:")
                
                correction_text = st.text_area(
                    "Correction Instructions",
                    placeholder="e.g., Sarah is on vacation next week, or Task T-101 is now high priority",
                    height=100
                )
                
                if st.button("🔄 Re-generate Plan with Corrections"):
                    if correction_text:
                        # Save correction to context
                        try:
                            st.session_state.db_handler.add_context(
                                f"CORRECTION: {correction_text}",
                                {"type": "correction", "timestamp": datetime.now().isoformat()}
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": f"Correction: {correction_text}"
                            })
                            
                            # Re-generate plan
                            with st.spinner("Re-generating plan with corrections..."):
                                combined_context = st.session_state.db_handler.get_combined_context()
                                st.session_state.model_config.update_from_session_state(st.session_state)
                                scrum_agents = CognitiveScrumAgents(st.session_state.model_config)
                                crew, scheduler_task, critic_task = scrum_agents.create_planning_crew(combined_context)
                                result = crew.kickoff()
                                
                                # Extract individual task outputs
                                try:
                                    scheduler_output = str(scheduler_task.output) if hasattr(scheduler_task, 'output') and scheduler_task.output else None
                                    critic_output = str(critic_task.output) if hasattr(critic_task, 'output') and critic_task.output else None
                                    
                                    # Fallback: if outputs not available, try to extract from crew tasks
                                    if not scheduler_output or not critic_output:
                                        if hasattr(crew, 'tasks') and len(crew.tasks) >= 3:
                                            scheduler_output = str(crew.tasks[1].output) if hasattr(crew.tasks[1], 'output') and crew.tasks[1].output else str(result)
                                            critic_output = str(crew.tasks[2].output) if hasattr(crew.tasks[2], 'output') and crew.tasks[2].output else str(result)
                                        else:
                                            scheduler_output = str(result)
                                            critic_output = str(result)
                                except Exception as e:
                                    scheduler_output = str(result)
                                    critic_output = str(result)
                                
                                # Parse scheduler output for table
                                task_df, _ = parse_sprint_plan_output(scheduler_output)
                                
                                # Store results
                                st.session_state.sprint_plan = critic_output
                                st.session_state.sprint_plan_table = task_df
                                st.session_state.sprint_plan_full_text = critic_output
                                st.success("✅ Plan updated with corrections!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error applying corrections: {str(e)}")
                    else:
                        st.warning("Please enter correction instructions.")


if __name__ == "__main__":
    main()
