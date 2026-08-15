import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import numpy as np
import seaborn as sns
from module.tail_classification.model import TailClassifier
from settings import BASE_DIR

def analyze_confidence_distribution():
    """Phân tích phân bố confidence scores cho từng class"""
    
    model_path = os.path.join(BASE_DIR, 'module', 'tail_classification', 'best_tail_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load model
    print("Loading model...")
    model = TailClassifier(model_path=model_path)
    print("Model loaded successfully!")
    
    # Get data paths
    speaking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "speaking")
    thinking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "thinking")
    
    # Collect confidence scores
    speaking_confidences = []
    thinking_confidences = []
    
    # Analyze speaking images
    if os.path.exists(speaking_dir):
        speaking_images = glob.glob(os.path.join(speaking_dir, "*.jpg"))
        print(f"Analyzing {len(speaking_images)} speaking images...")
        
        for i, img_path in enumerate(speaking_images):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(speaking_images)} speaking images")
            
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            pred_class, confidence = model.predict_single(image_array)
            
            if pred_class == 'speaking':
                speaking_confidences.append(confidence)
            else:
                # Model predicted thinking for a speaking image (wrong prediction)
                speaking_confidences.append(1.0 - confidence)  # Confidence in correct class
    
    # Analyze thinking images
    if os.path.exists(thinking_dir):
        thinking_images = glob.glob(os.path.join(thinking_dir, "*.jpg"))
        print(f"Analyzing {len(thinking_images)} thinking images...")
        
        for i, img_path in enumerate(thinking_images):
            if i % 20 == 0:
                print(f"  Processed {i}/{len(thinking_images)} thinking images")
            
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            pred_class, confidence = model.predict_single(image_array)
            
            if pred_class == 'thinking':
                thinking_confidences.append(confidence)
            else:
                # Model predicted speaking for a thinking image (wrong prediction)
                thinking_confidences.append(1.0 - confidence)  # Confidence in correct class
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram comparison
    axes[0, 0].hist(speaking_confidences, bins=30, alpha=0.7, label='Speaking', color='blue', density=True)
    axes[0, 0].hist(thinking_confidences, bins=30, alpha=0.7, label='Thinking', color='red', density=True)
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Confidence Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    data_to_plot = [speaking_confidences, thinking_confidences]
    axes[0, 1].boxplot(data_to_plot, labels=['Speaking', 'Thinking'])
    axes[0, 1].set_ylabel('Confidence Score')
    axes[0, 1].set_title('Confidence Score Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics
    speaking_mean = np.mean(speaking_confidences)
    thinking_mean = np.mean(thinking_confidences)
    speaking_std = np.std(speaking_confidences)
    thinking_std = np.std(thinking_confidences)
    
    # Bar chart of means
    classes = ['Speaking', 'Thinking']
    means = [speaking_mean, thinking_mean]
    stds = [speaking_std, thinking_std]
    
    axes[1, 0].bar(classes, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Mean Confidence Score')
    axes[1, 0].set_title('Mean Confidence by Class')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Violin plot
    data_for_violin = []
    data_for_violin.extend([('Speaking', conf) for conf in speaking_confidences])
    data_for_violin.extend([('Thinking', conf) for conf in thinking_confidences])
    
    import pandas as pd
    df = pd.DataFrame(data_for_violin, columns=['Class', 'Confidence'])
    
    sns.violinplot(data=df, x='Class', y='Confidence', ax=axes[1, 1])
    axes[1, 1].set_title('Confidence Distribution (Violin Plot)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS RESULTS")
    print("="*60)
    print(f"Speaking Class:")
    print(f"  Mean confidence: {speaking_mean:.4f}")
    print(f"  Std confidence:  {speaking_std:.4f}")
    print(f"  Min confidence:  {min(speaking_confidences):.4f}")
    print(f"  Max confidence:  {max(speaking_confidences):.4f}")
    print(f"  Samples:         {len(speaking_confidences)}")
    
    print(f"\nThinking Class:")
    print(f"  Mean confidence: {thinking_mean:.4f}")
    print(f"  Std confidence:  {thinking_std:.4f}")
    print(f"  Min confidence:  {min(thinking_confidences):.4f}")
    print(f"  Max confidence:  {max(thinking_confidences):.4f}")
    print(f"  Samples:         {len(thinking_confidences)}")
    
    print(f"\nDifference (Speaking - Thinking): {speaking_mean - thinking_mean:.4f}")
    
    # Confidence threshold analysis
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nConfidence Threshold Analysis:")
    print(f"{'Threshold':<12} {'Speaking %':<12} {'Thinking %':<12}")
    print("-" * 40)
    
    for threshold in thresholds:
        speaking_above = sum(1 for c in speaking_confidences if c >= threshold) / len(speaking_confidences) * 100
        thinking_above = sum(1 for c in thinking_confidences if c >= threshold) / len(thinking_confidences) * 100
        print(f"{threshold:<12.1f} {speaking_above:<12.1f} {thinking_above:<12.1f}")

def analyze_misclassifications():
    """Phân tích các trường hợp misclassification"""
    
    model_path = os.path.join(BASE_DIR, 'module', 'tail_classification', 'best_tail_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load model
    print("Loading model...")
    model = TailClassifier(model_path=model_path)
    print("Model loaded successfully!")
    
    # Get data paths
    speaking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "speaking")
    thinking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "thinking")
    
    misclassifications = []
    
    # Check speaking images
    if os.path.exists(speaking_dir):
        speaking_images = glob.glob(os.path.join(speaking_dir, "*.jpg"))
        for img_path in speaking_images[:50]:  # Check first 50 for speed
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            pred_class, confidence = model.predict_single(image_array)
            
            if pred_class != 'speaking':
                misclassifications.append({
                    'image_path': img_path,
                    'true_class': 'speaking',
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'correct_confidence': 1.0 - confidence
                })
    
    # Check thinking images
    if os.path.exists(thinking_dir):
        thinking_images = glob.glob(os.path.join(thinking_dir, "*.jpg"))
        for img_path in thinking_images:
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            pred_class, confidence = model.predict_single(image_array)
            
            if pred_class != 'thinking':
                misclassifications.append({
                    'image_path': img_path,
                    'true_class': 'thinking',
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'correct_confidence': 1.0 - confidence
                })
    
    print(f"\nMisclassifications found: {len(misclassifications)}")
    if misclassifications:
        print("\nTop 10 misclassifications:")
        print("-" * 80)
        sorted_mis = sorted(misclassifications, key=lambda x: x['correct_confidence'])
        for i, mis in enumerate(sorted_mis[:10]):
            print(f"{i+1:2d}. {os.path.basename(mis['image_path']):<30} | "
                  f"True: {mis['true_class']:<8} | Pred: {mis['pred_class']:<8} | "
                  f"Conf: {mis['correct_confidence']:.4f}")

def main():
    print("=== Confidence Analysis ===")
    analyze_confidence_distribution()
    print("\n=== Misclassification Analysis ===")
    analyze_misclassifications()

if __name__ == "__main__":
    main()
