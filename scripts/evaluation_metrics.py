#!/usr/bin/env python3
"""
Evaluation metrics cho speaker assignment và text extraction trong manga analysis system

Các metrics bao gồm:
1. Speaker Assignment Evaluation
2. Text Accuracy Assessment  
3. End-to-end Performance Analysis
"""

import json
import os
from pprint import pprint
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import difflib
from settings import BASE_DIR
from type import Transcript


@dataclass
class EvaluationResults:
    """Stores kết quả evaluation metrics"""
    speaker_accuracy: float = 0.0
    speaker_precision: float = 0.0
    speaker_recall: float = 0.0
    speaker_f1: float = 0.0
    text_recognition_accuracy: float = 0.0
    text_bleu_score: float = 0.0
    speech_type_accuracy: float = 0.0
    speech_type_per_label_accuracy: Dict | None = None
    speaker_confusion_matrix: Dict = None
    error_analysis: Dict = None


class SpeakerAssignmentEvaluator:
    """Evaluator cho speaker assignment accuracy"""
    
    def __init__(self):
        self.similar_speaker_mappings = {
            # Mapping các tên gần giống nhau
            "unknown": ["unsure", "other", "unknown"],
            "dư_lạc": ["du lac", "dư lạc", "yu le"],
            "tuyết_ly": ["tuyet ly", "tuyết ly", "xu lei"],
        }
    
    def normalize_speaker_name(self, speaker: str) -> str:
        """Normalize speaker name để so sánh dễ dàng hơn"""
        if not speaker:
            return "unknown"
        
        # Convert to lowercase và remove special characters
        normalized = re.sub(r'[^\w\s]', '', speaker.lower().strip())
        normalized = re.sub(r'\s+', '_', normalized)  # Replace spaces with underscore
        
        # Apply known mappings
        for canonical, aliases in self.similar_speaker_mappings.items():
            if normalized in [alias.lower() for alias in aliases]:
                return canonical
            
        return normalized
    
    def evaluate_speaker_assignment(
        self, 
        predicted_transcripts: List[Transcript], 
        ground_truth_transcripts: List[Transcript]
    ) -> Dict:
        """
        Đánh giá accuracy của speaker assignment
        
        Args:
            predicted_transcripts: Prediction results từ hệ thống
            ground_truth_transcripts: Ground truth annotations
            
        Returns:
            Dict với các metrics accuracy, precision, recall, f1
        """
        
        # Align predicted và ground truth transcripts
        aligned_pairs = self._align_transcripts(predicted_transcripts, ground_truth_transcripts)
        
        if len(aligned_pairs) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Extract labels for classification metrics
        pred_speakers = [self.normalize_speaker_name(pred.speaker) 
                        for pred, _ in aligned_pairs]
        gt_speakers = [self.normalize_speaker_name(gt.speaker) 
                      for _, gt in aligned_pairs]
        
        # Calculate metrics
        accuracy = accuracy_score(gt_speakers, pred_speakers)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_speakers, pred_speakers, average='weighted', zero_division=0
        )
        
        # Confusion matrix for detailed analysis
        confusion_matrix = self._build_confusion_matrix(gt_speakers, pred_speakers)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall, 
            "f1_score": f1,
            "confusion_matrix": confusion_matrix,
            "num_samples": len(aligned_pairs)
        }
    
    def _align_transcripts(
        self, 
        predicted: List[Transcript], 
        ground_truth: List[Transcript]
    ) -> List[Tuple[Transcript, Transcript]]:
        """
        Align predicted và ground truth transcripts bằng text similarity
        """
        aligned_pairs = []
        matched_gt_indices = set()
        
        for pred in predicted:
            best_match_idx = -1
            best_similarity = 0
            best_gt = None
            
            for i, gt in enumerate(ground_truth):
                if i in matched_gt_indices:
                    continue
                    
                similarity = self._calculate_text_similarity(pred.text, gt.text)
                if similarity > best_similarity and similarity > 0.3:  # Threshold
                    best_similarity = similarity
                    best_match_idx = i
                    best_gt = gt
            
            if best_match_idx != -1:
                aligned_pairs.append((pred, best_gt))
                matched_gt_indices.add(best_match_idx)
        
        return aligned_pairs
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity giữa 2 texts sử dụng difflib"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower())
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, text1_clean, text2_clean)
        return matcher.ratio()
    
    def _build_confusion_matrix(self, gt_labels: List[str], pred_labels: List[str]) -> Dict:
        """Build confusion matrix để analyze speaker assignment errors"""
        unique_labels = list(set(gt_labels + pred_labels))
        matrix = defaultdict(lambda: defaultdict(int))
        
        for gt, pred in zip(gt_labels, pred_labels):
            matrix[gt][pred] += 1
        
        # Convert to dict format
        result = {}
        for gt_label in unique_labels:
            result[gt_label] = dict(matrix[gt_label])
        
        return result


class TextAccuracyEvaluator:
    """Evaluator cho text recognition accuracy"""
    
    def evaluate_text_recognition(
        self, 
        predicted_transcripts: List[Transcript],
        ground_truth_transcripts: List[Transcript]
    ) -> Dict:
        """
        Đánh giá text recognition accuracy
        
        Args:
            predicted_transcripts: System predictions
            ground_truth_transcripts: True annotations
            
        Returns:
            Dict với text accuracy metrics
        """
        
        aligned_pairs = self._align_transcripts_text_only(
            predicted_transcripts, 
            ground_truth_transcripts
        )
        
        if len(aligned_pairs) == 0:
            return {"accuracy": 0.0, "bleu_score": 0.0, "character_accuracy": 0.0}
        
        # Calculate BLEU score approximation
        bleu_scores = []
        character_accuracies = []
        
        for pred_gt_pair in aligned_pairs:
            pred_text, gt_text = pred_gt_pair
            
            # BLEU-like score using n-grams
            bleu_score = self._calculate_bleu_approximation(pred_text, gt_text)
            bleu_scores.append(bleu_score)
            
            # Character-level accuracy
            char_accuracy = self._calculate_character_accuracy(pred_text, gt_text)
            character_accuracies.append(char_accuracy)
        
        return {
            "num_samples": len(aligned_pairs),
            "average_bleu_score": np.mean(bleu_scores),
            "average_character_accuracy": np.mean(character_accuracies),
            "bleu_scores": bleu_scores,
            "character_accuracies": character_accuracies
        }
    
    def _align_transcripts_text_only(
        self, 
        predicted: List[Transcript], 
        ground_truth: List[Transcript]
    ) -> List[Tuple[str, str]]:
        """Align transcripts based on text similarity only"""
        aligned_pairs = []
        
        for pred in predicted:
            best_match_similarity = 0
            best_gt_text = None
            
            for gt in ground_truth:
                similarity = self._text_similarity(pred.text, gt.text)
                if similarity > best_match_similarity and similarity > 0.2:
                    best_match_similarity = similarity
                    best_gt_text = gt.text
            
            if best_gt_text:
                aligned_pairs.append((pred.text, best_gt_text))
        
        return aligned_pairs
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using difflib"""
        matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    def _calculate_bleu_approximation(self, pred: str, gt: str) -> float:
        """Approximate BLEU score for text evaluation"""
        if not pred or not gt:
            return 0.0
        
        # Tokenize into words
        pred_tokens = pred.lower().split()
        gt_tokens = gt.lower().split()
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        # Simple BLEU approximation: precision of 4-gram matches
        overlaps = 0
        total_pred_tokens = 0
        
        for i in range(min(len(pred_tokens), len(gt_tokens))):
            if i < len(pred_tokens) and i < len(gt_tokens):
                if pred_tokens[i] == gt_tokens[i]:
                    overlaps += 1
            total_pred_tokens += 1
        
        precision = overlaps / max(len(pred_tokens), 1)
        
        # Brevity penalty
        if len(pred_tokens) > len(gt_tokens):
            bp = len(gt_tokens) / len(pred_tokens)
        else:
            bp = 1.0
        
        return precision * bp
    
    def _calculate_character_accuracy(self, pred: str, gt: str) -> float:
        """Calculate character-level accuracy"""
        if not pred or not gt:
            return 0.0
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, pred.lower(), gt.lower())
        matching_chars = matcher.get_matching_blocks()
        
        total_matches = sum(block.size for block in matching_chars)
        return total_matches / max(len(pred), len(gt), 1)


class ComprehensiveEvaluator:
    """Combined evaluator for speaker + text + end-to-end metrics"""
    
    def __init__(self):
        self.speaker_evaluator = SpeakerAssignmentEvaluator()
        self.text_evaluator = TextAccuracyEvaluator()
        self.speech_type_labels = ["speaking", "thinking"]
    
    def evaluate_transcript_system(
        self, 
        predictions: List[Transcript], 
        ground_truth: List[Transcript]
    ) -> EvaluationResults:
        """
        Comprehensive evaluation của transcript system
        
        Returns EvaluationResults với tất cả relevant metrics
        """
        
        # Evaluate speaker assignment
        speaker_results = self.speaker_evaluator.evaluate_speaker_assignment(
            predictions, ground_truth
        )
        
        # Evaluate text accuracy  
        text_results = self.text_evaluator.evaluate_text_recognition(
            predictions, ground_truth
        )
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions, ground_truth)
        
        # Evaluate speech type if available in both pred and gt
        speech_type_acc, per_label_acc = self._evaluate_speech_type(predictions, ground_truth)
        
        return EvaluationResults(
            speaker_accuracy=speaker_results.get("accuracy", 0.0),
            speaker_precision=speaker_results.get("precision", 0.0),
            speaker_recall=speaker_results.get("recall", 0.0),
            speaker_f1=speaker_results.get("f1_score", 0.0),
            text_recognition_accuracy=text_results.get("average_character_accuracy", 0.0),
            text_bleu_score=text_results.get("average_bleu_score", 0.0),
            speech_type_accuracy=speech_type_acc,
            speech_type_per_label_accuracy=per_label_acc,
            speaker_confusion_matrix=speaker_results.get("confusion_matrix"),
            error_analysis=error_analysis
        )
    
    def _analyze_errors(self, predictions: List, ground_truth: List) -> Dict:
        """Analyze specific error patterns để improve system"""
        errors = {
            "speaker_confusion": {},
            "text_errors": {},
            "missing_detections": 0,
            "false_positives": 0
        }
        
        # TODO: Implement detailed error analysis
        return errors
    
    def _evaluate_speech_type(self, predictions: List[Transcript], ground_truth: List[Transcript]) -> Tuple[float, Dict[str, float]]:
        """Đánh giá độ chính xác speaking/thinking bằng cách align theo text giống phần speaker."""
        # Align by text
        aligned_pairs = self.speaker_evaluator._align_transcripts(predictions, ground_truth)
        if len(aligned_pairs) == 0:
            return 0.0, {label: 0.0 for label in self.speech_type_labels}
        pred_labels = []
        gt_labels = []
        for pred, gt in aligned_pairs:
            pred_label = (pred.text_speech_type or "").lower()
            gt_label = (gt.text_speech_type or "").lower()
            if pred_label not in self.speech_type_labels or gt_label not in self.speech_type_labels:
                # skip pairs without valid labels
                continue
            pred_labels.append(pred_label)
            gt_labels.append(gt_label)
        if len(gt_labels) == 0:
            return 0.0, {label: 0.0 for label in self.speech_type_labels}
        overall_acc = accuracy_score(gt_labels, pred_labels)
        # Per-label accuracy
        per_label_acc = {}
        for label in self.speech_type_labels:
            indices = [i for i, g in enumerate(gt_labels) if g == label]
            if len(indices) == 0:
                per_label_acc[label] = 0.0
                continue
            label_gt = [gt_labels[i] for i in indices]
            label_pred = [pred_labels[i] for i in indices]
            per_label_acc[label] = accuracy_score(label_gt, label_pred)
        return overall_acc, per_label_acc
    
    def evaluate_from_json_files(
        self, 
        prediction_file_path: str, 
        ground_truth_file_path: str
    ) -> EvaluationResults:
        """Evaluate từ JSON files"""
        
        pred_transcripts = self._load_transcripts_from_file(prediction_file_path)
        gt_transcripts = self._load_transcripts_from_file(ground_truth_file_path)
        
        return self.evaluate_transcript_system(pred_transcripts, gt_transcripts)
    
    def _load_transcripts_from_file(self, file_path: str) -> List[Transcript]:
        """Load transcripts từ JSON file (hỗ trợ cả prediction & ground truth format)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        transcripts = []

        if "transcript" in data:
            # Prediction format
            for entry in data["transcript"]:
                transcripts.append(Transcript(
                    speaker=entry.get("speaker", "unknown"),
                    target=entry.get("target"),
                    text=entry.get("text", ""),
                    text_speech_type=entry.get("text_speech_type"),
                    translation=entry.get("translation")
                ))

        else:
            # Ground truth format: { "filename.json": [ {speaker, target, text}, ...] }
            for _, entries in data.items():
                for entry in entries:
                    transcripts.append(Transcript(
                        speaker=entry.get("speaker", "unknown"),
                        target=entry.get("target"),
                        text=entry.get("text", ""),
                        text_speech_type=entry.get("text_speech_type"),
                        translation=entry.get("translation")
                    ))

        return transcripts

def evaluate_from_test_file(test_file: str, base_dir: str, visualize: bool = True) -> Dict:
    """
    Evaluate nhiều trang từ 1 file test gộp.
    
    Args:
        test_file: đường dẫn file test (dạng { "path/to/file.json": [entries...] })
        base_dir: thư mục gốc chứa prediction files
        visualize: có vẽ biểu đồ hay không
    """
    evaluator = ComprehensiveEvaluator()
    all_results = {}

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # --- Evaluate từng file ---
    for relative_path, gt_entries in test_data.items():
        pred_file_path = os.path.join(base_dir, relative_path)

        if not os.path.exists(pred_file_path):
            all_results[relative_path] = {"error": f"Prediction file not found: {pred_file_path}"}
            continue

        # Convert ground truth entries thành Transcript list
        gt_transcripts = [
            Transcript(
                speaker=e.get("speaker", "unknown"),
                target=e.get("target"),
                text=e.get("text", ""),
                text_speech_type=e.get("text_speech_type")
            )
            for e in gt_entries
        ]

        # Load prediction
        pred_transcripts = evaluator._load_transcripts_from_file(pred_file_path)

        # Evaluate
        results = evaluator.evaluate_transcript_system(pred_transcripts, gt_transcripts)

        all_results[relative_path] = {
            "speaker_accuracy": results.speaker_accuracy,
            "text_bleu_score": results.text_bleu_score,
            "char_accuracy": results.text_recognition_accuracy,
            "speech_type_accuracy": results.speech_type_accuracy,
            "speech_type_per_label_accuracy": results.speech_type_per_label_accuracy,
            "overall_performance": (results.speaker_accuracy + results.text_bleu_score) / 2,
            "confusion_matrix": results.speaker_confusion_matrix
        }

    # --- Tính trung bình ---
    valid_results = [r for r in all_results.values() if "error" not in r]
    if not valid_results:
        return {"error": "No valid results for evaluation"}

    avg_speaker = np.mean([r["speaker_accuracy"] for r in valid_results])
    avg_bleu = np.mean([r["text_bleu_score"] for r in valid_results])
    avg_char = np.mean([r["char_accuracy"] for r in valid_results])
    avg_speech_type = np.mean([r["speech_type_accuracy"] for r in valid_results])

    # Per-label averages for speech type
    per_label_list = [r.get("speech_type_per_label_accuracy") for r in valid_results]
    per_label_list = [d for d in per_label_list if isinstance(d, dict) and len(d) > 0]
    if len(per_label_list) > 0:
        speaking_vals = [d.get("speaking") for d in per_label_list if d.get("speaking") is not None]
        thinking_vals = [d.get("thinking") for d in per_label_list if d.get("thinking") is not None]
        avg_speaking = float(np.mean(speaking_vals)) if len(speaking_vals) > 0 else 0.0
        avg_thinking = float(np.mean(thinking_vals)) if len(thinking_vals) > 0 else 0.0
    else:
        avg_speaking = 0.0
        avg_thinking = 0.0

    summary = {
        "avg_speaker_acc": avg_speaker,
        "avg_text_bleu": avg_bleu,
        "avg_char_acc": avg_char,
        "avg_speech_type_acc": avg_speech_type,
        "avg_speaking_acc": avg_speaking,
        "avg_thinking_acc": avg_thinking,
        "total_pages": len(valid_results)
    }   

    pprint(summary, indent=4)

    # --- Visualization ---
    if visualize:
        import matplotlib.pyplot as plt

        valid_items = [(fname, r) for fname, r in all_results.items() if "error" not in r]

        files = [fname for fname, _ in valid_items]
        speaker_accs = [r["speaker_accuracy"] for _, r in valid_items]
        bleu_scores = [r["text_bleu_score"] for _, r in valid_items]
        char_accs = [r["char_accuracy"] for _, r in valid_items]
        speech_type_accs = [r["speech_type_accuracy"] for _, r in valid_items]

        x = range(len(files))

        plt.figure(figsize=(12, 6))
        plt.plot(x, speaker_accs, marker='o', label="Speaker Accuracy")
        plt.plot(x, bleu_scores, marker='s', label="Text BLEU")
        plt.plot(x, char_accs, marker='^', label="Character Accuracy")
        plt.plot(x, speech_type_accs, marker='*', label="Speech Type Accuracy")
        
        plt.axhline(y=avg_speaker, color='blue', linestyle='--', alpha=0.5, label="Avg Speaker")
        plt.axhline(y=avg_bleu, color='orange', linestyle='--', alpha=0.5, label="Avg BLEU")
        plt.axhline(y=avg_char, color='green', linestyle='--', alpha=0.5, label="Avg Char Acc")
        plt.axhline(y=avg_speech_type, color='purple', linestyle='--', alpha=0.5, label="Avg Speech Type Acc")
        
        plt.xticks(x, files, rotation=45, ha="right")
        plt.ylabel("Score")
        plt.title("Evaluation Metrics per Page / Episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig("evaluation_metrics.png")
        plt.show()


    return {"per_page": all_results, "summary": summary}

   


def evaluate_whole_chapter(pred_dir: str, gt_dir: str) -> Dict:
    """Evaluate cả chapter với multiple pages"""
    evaluator = ComprehensiveEvaluator()
    all_speaker_accuracies = []
    all_text_scores = []
    
    # Find all JSON files to compare
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    
    for pred_file in pred_files:
        if pred_file in gt_files:
            pred_path = os.path.join(pred_dir, pred_file)
            gt_path = os.path.join(gt_dir, pred_file)
            
            results = evaluator.evaluate_from_json_files(pred_path, gt_path)
            all_speaker_accuracies.append(results.speaker_accuracy)
            all_text_scores.append(results.text_bleu_score)
    
    if len(all_speaker_accuracies) == 0:
        return {"error": "No matching files found for evaluation"}
    
    return {
        "chapter_avg_speaker_accuracy": np.mean(all_speaker_accuracies),
        "chapter_avg_text_bleu": np.mean(all_text_scores),
        "chapter_speaker_std": np.std(all_speaker_accuracies),
        "chapter_text_std": np.std(all_text_scores),
        "total_pages_evaluated": len(all_speaker_accuracies)
    }


if __name__ == "__main__":
    TEST_FILE = "/home/aorus/workspaces/magiv2/test_data/test_transcripts_133.json"

    results = evaluate_from_test_file(TEST_FILE, BASE_DIR, visualize=True)

    print("\n📊 Summary:")
    pprint(results["summary"], indent=4)