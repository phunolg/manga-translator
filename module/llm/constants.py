from enum import Enum


class LLM_MODEL(str, Enum):
    chatgpt = "chatgpt"
    gemini = "gemini"
    gemma3 = "gemma3"