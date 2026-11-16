"""Configuration and API client helpers for Day 7 - Agentic RAG."""
import os
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def _clean_key(val: str | None) -> str:
    if not val:
        return ""
    val = val.strip()
    if val.startswith("your_"):
        return ""
    return val

# OpenAI Settings (for Chat Completion / Agent Controller)
OPENAI_API_KEY = _clean_key(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
OPENAI_BASE_URL = _clean_key(os.getenv("OPENAI_BASE_URL")) or None

# Google Gemini Settings (for Embedding Model)
GOOGLE_API_KEY = _clean_key(os.getenv("GEMINI_API_KEY")) or _clean_key(os.getenv("GOOGLE_API_KEY"))
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004").strip() or "text-embedding-004"


def get_openai_client() -> Any:
    """Return an initialized OpenAI client.
    
    Supports custom base_url (via OPENAI_BASE_URL).
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        raise ValueError(
            "OPENAI_API_KEY chưa được thiết lập. "
            "Vui lòng điền API key vào file .env hoặc export vào môi trường."
        )
    from openai import OpenAI
    kwargs: dict[str, Any] = {
        "api_key": OPENAI_API_KEY,
        "timeout": 30.0,
        "max_retries": 2,
    }
    if OPENAI_BASE_URL and not OPENAI_BASE_URL.startswith("your_"):
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def get_google_genai_client() -> Any:
    """Return an initialized Google GenAI client.
    
    Raises:
        ValueError: If GOOGLE_API_KEY or GEMINI_API_KEY is not set.
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("your_"):
        raise ValueError(
            "GOOGLE_API_KEY hoặc GEMINI_API_KEY chưa được thiết lập. "
            "Vui lòng điền API key vào file .env hoặc export vào môi trường."
        )
    try:
        from google import genai
        return genai.Client(api_key=GOOGLE_API_KEY)
    except Exception:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=GOOGLE_API_KEY)
        return genai_legacy
