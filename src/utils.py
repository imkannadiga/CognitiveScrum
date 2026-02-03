"""
Utility functions for PDF parsing and data processing.
"""
import PyPDF2
import pandas as pd
from typing import List, Dict, Optional
import io


def parse_pdf(file) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file: File-like object or file path
        
    Returns:
        Extracted text string
    """
    try:
        # Handle Streamlit UploadedFile
        if hasattr(file, 'read'):
            file.seek(0)
            reader = PyPDF2.PdfReader(file)
        else:
            # Assume it's a file path
            reader = PyPDF2.PdfReader(file)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error parsing PDF: {str(e)}")


def parse_csv_backlog(file) -> List[Dict]:
    """
    Parse CSV backlog file into list of dictionaries.
    
    Expected columns: ticket_id, description, complexity, required_skills
    
    Args:
        file: CSV file (Streamlit UploadedFile or path)
        
    Returns:
        List of backlog item dictionaries
    """
    try:
        # Handle Streamlit UploadedFile
        if hasattr(file, 'read'):
            file.seek(0)
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
        
        backlog_items = []
        for _, row in df.iterrows():
            item = {
                "ticket_id": str(row.get("ticket_id", "")),
                "description": str(row.get("description", "")),
                "complexity": str(row.get("complexity", "Medium")),
                "required_skills": str(row.get("required_skills", ""))
            }
            backlog_items.append(item)
        
        return backlog_items
    except Exception as e:
        raise Exception(f"Error parsing CSV: {str(e)}")


def parse_json_backlog(file) -> List[Dict]:
    """
    Parse JSON backlog file into list of dictionaries.
    
    Args:
        file: JSON file (Streamlit UploadedFile or path)
        
    Returns:
        List of backlog item dictionaries
    """
    try:
        import json
        
        # Handle Streamlit UploadedFile
        if hasattr(file, 'read'):
            file.seek(0)
            data = json.load(file)
        else:
            with open(file, 'r') as f:
                data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'backlog' in data:
            return data['backlog']
        else:
            return [data]
    except Exception as e:
        raise Exception(f"Error parsing JSON: {str(e)}")


def extract_candidate_name(resume_text: str) -> str:
    """
    Simple heuristic to extract candidate name from resume.
    Usually the first line or first few words.
    
    Args:
        resume_text: Full resume text
        
    Returns:
        Candidate name (or "Unknown")
    """
    lines = resume_text.strip().split('\n')
    if lines:
        # First non-empty line is often the name
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4:  # Name is usually short
                return line
    return "Unknown"


def parse_sprint_plan_output(plan_text: str) -> tuple:
    """
    Parse CrewAI sprint plan output to extract structured task assignments.
    
    Args:
        plan_text: Full text output from CrewAI crew
        
    Returns:
        Tuple of (DataFrame with task assignments, full text response)
    """
    import re
    
    # Try to extract table data from the text
    task_assignments = []
    
    # Look for table-like structures with Task_ID, Assignee, etc.
    # Pattern 1: Markdown table format
    markdown_table_pattern = r'\|.*Task.*\|.*Assignee.*\|.*Hours.*\|.*Risk.*\|'
    
    # Pattern 2: Structured text format with Task_ID, Assignee, etc.
    # Handle multi-line reasoning traces that continue after table rows
    lines = plan_text.split('\n')
    in_table_section = False
    table_row_indices = []  # Store (line_content, line_number) for table rows
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        line_stripped = line.strip()
        
        # Detect table start - look for header row
        if any(keyword in line_lower for keyword in ['task_id', 'assignee', 'estimated_hours', 'risk_level', 'reasoning_trace']):
            if '|' in line:
                in_table_section = True
                table_row_indices.append((line, i))
                continue
        
        if in_table_section:
            if '|' in line and any(char.isalnum() for char in line):  # Table row with content
                table_row_indices.append((line, i))
            elif not line_stripped:
                # Empty line - continue
                pass
            elif '|' not in line and i > 0:
                # Check if this is reasoning continuation (within 5 lines of last table row)
                if table_row_indices:
                    last_row_idx = table_row_indices[-1][1]
                    if i - last_row_idx <= 5:
                        # Might be reasoning continuation - check for keywords
                        if any(keyword in line_stripped.lower() for keyword in ['assignment:', 'estimate:', 'risk:', '**']):
                            # This is reasoning continuation, attach to last row
                            last_line, last_idx = table_row_indices[-1]
                            table_row_indices[-1] = (last_line + " " + line_stripped, last_idx)
                        elif i - last_row_idx <= 2:
                            # Very close, likely continuation
                            last_line, last_idx = table_row_indices[-1]
                            table_row_indices[-1] = (last_line + " " + line_stripped, last_idx)
                    else:
                        # Too far, end table section
                        in_table_section = False
                else:
                    in_table_section = False
            else:
                # Other content, end table section
                in_table_section = False
    
    # Try to parse as markdown table
    if table_row_indices:
        try:
            # Clean up markdown table lines
            cleaned_lines = []
            for line_data, _ in table_row_indices:
                if '|' in line_data:
                    # Split by | and clean
                    parts = [p.strip() for p in line_data.split('|')]
                    # Remove empty first/last elements from markdown table format
                    if parts and not parts[0]:
                        parts = parts[1:]
                    if parts and not parts[-1]:
                        parts = parts[:-1]
                    # Filter out separator rows (all dashes/colons)
                    if parts and len(parts) >= 3:
                        # Check if it's a separator row
                        is_separator = all(not p.strip() or all(c in '-: ' for c in p.strip()) for p in parts)
                        if not is_separator:
                            cleaned_lines.append(parts)
            
            if cleaned_lines and len(cleaned_lines) > 1:
                # First line is header
                headers = cleaned_lines[0]
                # Find relevant column indices
                task_col = None
                assignee_col = None
                hours_col = None
                risk_col = None
                reasoning_col = None
                
                for idx, header in enumerate(headers):
                    header_lower = header.lower()
                    if 'task' in header_lower and task_col is None:
                        task_col = idx
                    if 'assign' in header_lower and assignee_col is None:
                        assignee_col = idx
                    if 'hour' in header_lower or 'time' in header_lower or 'estimat' in header_lower:
                        hours_col = idx
                    if 'risk' in header_lower:
                        risk_col = idx
                    if 'reason' in header_lower or 'trace' in header_lower:
                        reasoning_col = idx
                
                # Extract data rows - handle multi-line reasoning traces
                lines_list = plan_text.split('\n')
                row_idx = 0
                
                for row in cleaned_lines[1:]:  # Skip header
                    # Skip separator lines
                    if all(part.strip().replace('-', '').replace(':', '').strip() == '' for part in row):
                        continue
                    
                    # Ensure we have enough columns
                    max_col_idx = max([idx for idx in [task_col, assignee_col, hours_col, risk_col, reasoning_col] if idx is not None], default=-1)
                    if len(row) > max_col_idx:
                        # Extract basic fields
                        task_id_val = row[task_col] if task_col is not None and task_col < len(row) else "N/A"
                        assignee_val = row[assignee_col] if assignee_col is not None and assignee_col < len(row) else "N/A"
                        hours_val = row[hours_col] if hours_col is not None and hours_col < len(row) else "N/A"
                        risk_val = row[risk_col] if risk_col is not None and risk_col < len(row) else "N/A"
                        
                        # Extract reasoning trace - might span multiple lines
                        reasoning_text = ""
                        if reasoning_col is not None and reasoning_col < len(row):
                            reasoning_text = row[reasoning_col]
                        
                        # Look for continuation of reasoning in subsequent lines
                        # Find the line number of this row in original text
                        for line_num, orig_line in enumerate(lines_list):
                            if task_id_val in orig_line and '|' in orig_line and assignee_val[:10] in orig_line:
                                # Found the row, check next few lines for reasoning continuation
                                reasoning_lines = []
                                for next_line_num in range(line_num + 1, min(line_num + 15, len(lines_list))):
                                    next_line = lines_list[next_line_num].strip()
                                    # Stop if we hit another table row with task ID pattern
                                    if '|' in next_line:
                                        # Check if it's a new task row (has task ID pattern)
                                        if any(pattern in next_line.lower() for pattern in ['s1-t', 's2-t', 's3-t', 's4-t', 's5-t', 't-']):
                                            break
                                    # Collect reasoning text
                                    if next_line and not next_line.startswith('|'):
                                        if any(keyword in next_line for keyword in ['**Assignment:**', '**Estimate:**', '**Risk:**', 'Assignment:', 'Estimate:', 'Risk:']):
                                            reasoning_lines.append(next_line)
                                        elif len(reasoning_lines) > 0 and next_line:  # Continue if we're already collecting
                                            # Stop if we hit a new section or too much text
                                            if len(reasoning_lines) > 5:  # Limit reasoning length
                                                break
                                            reasoning_lines.append(next_line)
                                        elif not reasoning_lines and len(next_line) > 50:  # Long line might be reasoning start
                                            reasoning_lines.append(next_line)
                                if reasoning_lines:
                                    reasoning_text = (reasoning_text + " " + " ".join(reasoning_lines)).strip()
                                break
                        
                        assignment = {
                            "Task_ID": task_id_val,
                            "Assignee": assignee_val,
                            "Estimated_Hours": hours_val,
                            "Risk_Level": risk_val,
                            "Reasoning_Trace": reasoning_text if reasoning_text else "See full report"
                        }
                        # Only add if we have at least Task_ID and Assignee
                        if assignment["Task_ID"] != "N/A" and assignment["Assignee"] != "N/A":
                            task_assignments.append(assignment)
        except Exception:
            pass
    
    # Alternative: Try to extract structured data from text patterns
    if not task_assignments:
        # Look for patterns like "Task T-101: Assignee: John, Hours: 8, Risk: Low"
        pattern = r'(?:Task[_\s]*(?:ID)?[:\s]*)?([T\-]?\d+|[A-Z]+\-\d+)[:\s]+(?:Assignee[:\s]+)?([A-Za-z\s]+?)(?:[,\s]+Hours?[:\s]+)?(\d+)?(?:[,\s]+Risk[:\s]+)?([A-Za-z]+)?'
        matches = re.finditer(pattern, plan_text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            task_id = match.group(1) if match.group(1) else "N/A"
            assignee = match.group(2).strip() if match.group(2) else "N/A"
            hours = match.group(3) if match.group(3) else "N/A"
            risk = match.group(4) if match.group(4) else "N/A"
            
            task_assignments.append({
                "Task_ID": task_id,
                "Assignee": assignee,
                "Estimated_Hours": hours,
                "Risk_Level": risk,
                "Reasoning_Trace": "See full report below"
            })
    
    # Create DataFrame
    if task_assignments:
        df = pd.DataFrame(task_assignments)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=["Task_ID", "Assignee", "Estimated_Hours", "Risk_Level", "Reasoning_Trace"])
    
    return df, plan_text
