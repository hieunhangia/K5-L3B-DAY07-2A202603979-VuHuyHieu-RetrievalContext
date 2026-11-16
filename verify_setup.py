"""Verify environment setup for Day 7 - Agentic RAG."""
import sys

def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    errors = []

    # Check OpenAI
    try:
        import openai
        print(f"[OK] openai version: {openai.__version__}")
    except ImportError as e:
        errors.append(f"[FAIL] openai import error: {e}")

    # Check Google GenAI / GenerativeAI
    try:
        import google.genai as genai
        print(f"[OK] google-genai imported successfully")
    except ImportError as e:
        try:
            import google.generativeai as genai_legacy
            print(f"[OK] google-generativeai version: {genai_legacy.__version__}")
        except ImportError as e2:
            errors.append(f"[FAIL] google genai import error: {e} | {e2}")

    # Check rank_bm25
    try:
        import rank_bm25
        print(f"[OK] rank_bm25 imported successfully")
    except ImportError as e:
        errors.append(f"[FAIL] rank_bm25 import error: {e}")

    # Check scikit-learn
    try:
        import sklearn
        print(f"[OK] scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        errors.append(f"[FAIL] scikit-learn import error: {e}")

    # Check numpy
    try:
        import numpy as np
        print(f"[OK] numpy version: {np.__version__}")
    except ImportError as e:
        errors.append(f"[FAIL] numpy import error: {e}")

    # Check python-dotenv
    try:
        import dotenv
        print(f"[OK] python-dotenv imported successfully")
    except ImportError as e:
        errors.append(f"[FAIL] python-dotenv import error: {e}")

    if errors:
        print("\nSetup verification FAILED:")
        for err in errors:
            print("  ", err)
        sys.exit(1)
    else:
        print("\n[SUCCESS] Tất cả thư viện phục vụ OpenAI Chat & Google Embedding đã sẵn sàng!")

if __name__ == "__main__":
    main()
