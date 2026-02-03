import PyPDF2
from pydantic import BaseModel, Field
from typing import List

# Define the structured output format (Data Normalization Layer)
class EmployeeProfile(BaseModel):
    name: str
    years_experience: int
    skills: List[str] = Field(description="List of technical skills extracted")
    seniority_level: str = Field(description="Junior, Mid, Senior, or Lead")

def parse_pdf(file_path):
    """
    Extracts raw text from a PDF resume.
    Mitigates: Unstructured Inputs challenge.
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Note: In the next step, we will connect this function to the CrewAI 
# agent so Llama 3 can process the raw text into the EmployeeProfile format.