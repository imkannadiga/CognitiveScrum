"""
ChromaDB Handler for Persistent Vector Storage.
Manages three collections: resumes, backlog, project_context.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os


class DBHandler:
    """Manages ChromaDB collections for the CognitiveScrum application."""
    
    def __init__(self, persist_directory: str = "./scrum_db"):
        """
        Initialize ChromaDB with persistent storage.
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize collections
        self.resumes_collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"description": "Stored resume data"}
        )
        
        self.backlog_collection = self.client.get_or_create_collection(
            name="backlog",
            metadata={"description": "Stored backlog items"}
        )
        
        self.project_context_collection = self.client.get_or_create_collection(
            name="project_context",
            metadata={"description": "Stored conversational context from interviews"}
        )
    
    def add_resume(self, text: str, metadata: Dict, candidate_id: Optional[str] = None):
        """
        Add resume text to the resumes collection.
        
        Args:
            text: Resume text content
            metadata: Dictionary with metadata (name, skills, etc.)
            candidate_id: Optional unique ID for the candidate
        """
        if candidate_id is None:
            candidate_id = f"candidate_{len(self.resumes_collection.get()['ids'])}"
        
        # Split text into chunks for better retrieval
        chunks = self._chunk_text(text, chunk_size=500)
        
        ids = [f"{candidate_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]
        
        self.resumes_collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        return candidate_id
    
    def add_backlog_item(self, text: str, metadata: Dict, item_id: Optional[str] = None):
        """
        Add backlog item to the backlog collection.
        
        Args:
            text: Backlog item description
            metadata: Dictionary with metadata (ticket_id, complexity, etc.)
            item_id: Optional unique ID for the item
        """
        if item_id is None:
            item_id = f"backlog_{len(self.backlog_collection.get()['ids'])}"
        
        self.backlog_collection.add(
            documents=[text],
            ids=[item_id],
            metadatas=[metadata]
        )
        
        return item_id
    
    def add_context(self, text: str, metadata: Optional[Dict] = None):
        """
        Add conversational context from the interview.
        
        Args:
            text: User answer or context text
            metadata: Optional metadata (question, timestamp, etc.)
        """
        if metadata is None:
            metadata = {}
        
        context_id = f"context_{len(self.project_context_collection.get()['ids'])}"
        
        self.project_context_collection.add(
            documents=[text],
            ids=[context_id],
            metadatas={**metadata, "type": "interview_response"}
        )
        
        return context_id
    
    def get_combined_context(self, query: str = "", n_results: int = 10) -> str:
        """
        Retrieve all relevant context from all collections.
        Used to feed the Planner Agent.
        
        Args:
            query: Query string for semantic search
            n_results: Number of results per collection
            
        Returns:
            Combined context string
        """
        context_parts = []
        
        # Get relevant resumes
        if query:
            resume_results = self.resumes_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if resume_results['documents']:
                context_parts.append("=== RESUME DATA ===")
                for doc, metadata in zip(resume_results['documents'][0], resume_results['metadatas'][0]):
                    name = metadata.get('name', 'Unknown')
                    context_parts.append(f"\nCandidate: {name}\n{doc}\n")
        else:
            # Get all resumes if no query
            all_resumes = self.resumes_collection.get()
            if all_resumes['documents']:
                context_parts.append("=== RESUME DATA ===")
                for doc, metadata in zip(all_resumes['documents'], all_resumes['metadatas']):
                    name = metadata.get('name', 'Unknown')
                    context_parts.append(f"\nCandidate: {name}\n{doc}\n")
        
        # Get relevant backlog items
        if query:
            backlog_results = self.backlog_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if backlog_results['documents']:
                context_parts.append("\n=== BACKLOG ITEMS ===")
                for doc, metadata in zip(backlog_results['documents'][0], backlog_results['metadatas'][0]):
                    ticket_id = metadata.get('ticket_id', 'Unknown')
                    context_parts.append(f"\nTicket: {ticket_id}\n{doc}\n")
        else:
            all_backlog = self.backlog_collection.get()
            if all_backlog['documents']:
                context_parts.append("\n=== BACKLOG ITEMS ===")
                for doc, metadata in zip(all_backlog['documents'], all_backlog['metadatas']):
                    ticket_id = metadata.get('ticket_id', 'Unknown')
                    context_parts.append(f"\nTicket: {ticket_id}\n{doc}\n")
        
        # Get project context (interview responses)
        if query:
            context_results = self.project_context_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if context_results['documents']:
                context_parts.append("\n=== PROJECT CONTEXT (INTERVIEW) ===")
                for doc in context_results['documents'][0]:
                    context_parts.append(f"\n{doc}\n")
        else:
            all_context = self.project_context_collection.get()
            if all_context['documents']:
                context_parts.append("\n=== PROJECT CONTEXT (INTERVIEW) ===")
                for doc in all_context['documents']:
                    context_parts.append(f"\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def get_all_resumes(self) -> List[Dict]:
        """Get all resumes with metadata."""
        results = self.resumes_collection.get()
        return [
            {
                "id": id,
                "text": doc,
                "metadata": meta
            }
            for id, doc, meta in zip(results['ids'], results['documents'], results['metadatas'])
        ]
    
    def get_all_backlog(self) -> List[Dict]:
        """Get all backlog items with metadata."""
        results = self.backlog_collection.get()
        return [
            {
                "id": id,
                "text": doc,
                "metadata": meta
            }
            for id, doc, meta in zip(results['ids'], results['documents'], results['metadatas'])
        ]
    
    def reset_db(self):
        """Clear all collections."""
        self.client.delete_collection("resumes")
        self.client.delete_collection("backlog")
        self.client.delete_collection("project_context")
        
        # Recreate collections
        self.resumes_collection = self.client.get_or_create_collection(name="resumes")
        self.backlog_collection = self.client.get_or_create_collection(name="backlog")
        self.project_context_collection = self.client.get_or_create_collection(name="project_context")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks for better vector storage."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]
