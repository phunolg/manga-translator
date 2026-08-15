from module.llm.llama_cpp import LlamaCpp
from settings import VLLM_BASE_URL, VLLM_API_KEY
from module.llm.constants import LLM_MODEL
from settings import GEMINI_API_KEY
def get_llm(model: LLM_MODEL):
    match model:
        case LLM_MODEL.gemma3:
            return LlamaCpp(VLLM_BASE_URL, VLLM_API_KEY, "google/gemma-3-27b-it-qat-q4_0-gguf:Q4_0")
        case LLM_MODEL.gemini:
            return LlamaCpp("https://generativelanguage.googleapis.com/v1beta/openai/", GEMINI_API_KEY, "gemini-2.5-flash")
        case LLM_MODEL.chatgpt:
            return LlamaCpp(VLLM_BASE_URL, VLLM_API_KEY, "google/gemma-3-27b-it-qat-q4_0-gguf:Q4_0")
        case _:
            raise ValueError(f"Model {model} not supported")
