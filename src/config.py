"""
Dynamic LLM Configuration Wrapper using LiteLLM.
Allows users to configure LLM settings from Streamlit UI.
"""
import os
from typing import Optional
import litellm

# Try different import paths for ChatLiteLLM
# Prefer langchain-litellm (new package), fallback to langchain_community
try:
    from langchain_litellm import ChatLiteLLM
except ImportError:
    try:
        from langchain_community.chat_models import ChatLiteLLM
    except ImportError:
        try:
            from langchain_community.llms import ChatLiteLLM
        except ImportError:
            ChatLiteLLM = None  # Will use fallback

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


class ModelConfig:
    """Manages LLM configuration dynamically from Streamlit session state."""
    
    def __init__(self):
        """Initialize with default values."""
        self.model_name = "llama3"
        self.api_key = None
        self.base_url = "http://localhost:11434"
        self._llm_instance = None
    
    def update_from_session_state(self, session_state):
        """Update configuration from Streamlit session state."""
        self.model_name = session_state.get("model_name", "llama3")
        self.api_key = session_state.get("api_key", None)
        self.base_url = session_state.get("base_url", "http://localhost:11434")
        # Reset LLM instance when config changes
        self._llm_instance = None

    @staticmethod
    def _normalize_model_id(model_name: str, base_url: Optional[str]) -> str:
        """
        Normalize user-entered model names into LiteLLM provider format.

        Supported examples:
        - llama3 + http://localhost:11434  -> ollama/llama3
        - gpt-4o                           -> openai/gpt-4o
        - claude-3-5-sonnet-latest         -> anthropic/claude-3-5-sonnet-latest
        - gemini-1.5-pro                   -> gemini/gemini-1.5-pro
        - gemini/gemini-1.5-flash          -> gemini/gemini-1.5-flash (kept)
        """
        mn = (model_name or "").strip()
        if not mn:
            mn = "llama3"

        # If user already provided provider/model, keep it.
        if "/" in mn:
            # Common Gemini pattern from Google docs: "models/gemini-1.5-pro"
            if mn.startswith("models/"):
                return f"gemini/{mn.replace('models/', '', 1)}"
            return mn

        bu = (base_url or "").strip()
        is_local = bu and ("localhost" in bu or "127.0.0.1" in bu)

        if is_local:
            return f"ollama/{mn}"
        if mn.startswith("gpt-") or mn.startswith("o1-") or mn.startswith("o3-"):
            return f"openai/{mn}"
        if mn.startswith("claude-"):
            return f"anthropic/{mn}"
        if mn.startswith("gemini"):
            return f"gemini/{mn}"

        # Default: assume user entered a valid LiteLLM model name
        return mn

    @staticmethod
    def _set_provider_env(provider: str, api_key: Optional[str], api_base: Optional[str]) -> None:
        """Set provider-specific env vars expected by LiteLLM."""
        if not api_key:
            return

        p = (provider or "").lower().strip()
        if p in {"openai"}:
            os.environ["OPENAI_API_KEY"] = api_key
            if api_base:
                os.environ["OPENAI_API_BASE"] = api_base
        elif p in {"anthropic"}:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif p in {"gemini", "google"}:
            # LiteLLM typically accepts either of these depending on backend/version.
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
        # VertexAI is credential-file based; API key is not enough.
    
    def get_llm(self):
        """
        Returns a LangChain LLM object compatible with CrewAI.
        Uses LiteLLM to abstract model calls.
        """
        # Return cached instance if config hasn't changed
        if self._llm_instance is not None:
            return self._llm_instance
        
        try:
            base_url = (self.base_url or "").strip() or None
            model_id = self._normalize_model_id(self.model_name, base_url)
            provider = model_id.split("/", 1)[0] if "/" in model_id else ""

            # Configure provider auth/env
            self._set_provider_env(provider, self.api_key, base_url)

            # Only set global api_base for local ollama; for cloud providers it can break routing.
            if provider == "ollama" and base_url:
                litellm.api_base = base_url
            
            # Create LangChain ChatLiteLLM wrapper
            # This is compatible with CrewAI
            if ChatLiteLLM:
                self._llm_instance = ChatLiteLLM(
                    model=model_id,
                    api_key=self.api_key if self.api_key else None,
                    # Only pass api_base for ollama/local usage
                    api_base=base_url if provider == "ollama" else None,
                    temperature=0.7
                )
            else:
                # Fallback: use litellm directly via OpenAI-compatible interface
                if ChatOpenAI and model_id.startswith("openai/"):
                    self._llm_instance = ChatOpenAI(
                        model_name=self.model_name,
                        openai_api_key=self.api_key,
                        temperature=0.7
                    )
                else:
                    raise Exception("ChatLiteLLM not available. Please install langchain-community.")
            
            return self._llm_instance
            
        except Exception as e:
            # Fallback: try using OpenAI-compatible interface
            if ChatOpenAI and (self.model_name or "").startswith("gpt-"):
                try:
                    self._llm_instance = ChatOpenAI(
                        model_name=self.model_name,
                        openai_api_key=self.api_key,
                        temperature=0.7
                    )
                    return self._llm_instance
                except:
                    pass
            
            raise Exception(f"Failed to initialize LLM: {str(e)}. Please check your model configuration.")
    
    def test_connection(self):
        """Test the LLM connection with a simple prompt."""
        try:
            llm = self.get_llm()
            # Simple test prompt
            test_prompt = "Say 'Connection successful' if you can read this."
            response = llm.invoke(test_prompt)
            return True, str(response.content) if hasattr(response, 'content') else str(response)
        except Exception as e:
            return False, str(e)
