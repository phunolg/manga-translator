import glob
import json
import os
transcript_paths = glob.glob("transcript_history/Yule/133/*.json")
transcript_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
test_transcripts = {}
for transcript_path in transcript_paths:
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    test_transcripts[transcript_path] = transcript["transcript"]

with open("test_transcripts.json", "w", encoding="utf-8") as f:
    json.dump(test_transcripts, f, ensure_ascii=False, indent=4)