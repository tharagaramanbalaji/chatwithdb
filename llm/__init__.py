from .gemini import GeminiManager
from .ollama import OllamaManager

def get_llm_manager(llm_type: str, model_name: str, **kwargs):
    llm_type = llm_type.lower().strip()
    if llm_type == "gemini":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Gemini requires an API key")
        return GeminiManager(api_key=api_key, model_name=model_name)
    elif llm_type == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaManager(base_url=base_url, model_name=model_name)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
