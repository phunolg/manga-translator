import json
from typing import List, Tuple
from type import Transcript
from module.history_retrival.type import PageInfo
class TranscriptRepository:
    def __init__(self):
        pass
    
    @staticmethod
    def get_transcript(page_info: PageInfo) -> Tuple[List[Transcript], List[List[int]]]:
        with open(page_info.get_save_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Transcript(**transcript) for transcript in data.get("transcript", [])], data["transcript_bboxes"]
        return [], []
    