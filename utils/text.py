import re
from typing import List
def classify_text(text: str, predict_type: str) -> str:
    if predict_type == "thinking":
        if "you" in text.lower():
            return "speaking"
        return "thinking"
    
    if predict_type == "speaking":
        if "?" in text or "!" in text:
            return "speaking"
        return "speaking"
