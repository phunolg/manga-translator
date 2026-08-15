import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import random
import numpy as np
from module.tail_classification.model import TailClassifier
from settings import BASE_DIR

def demo_inference(model_path: str, num_samples: int = 10):
    """Demo inference on random samples"""
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize inference
    print("Loading model...")
    inference = TailClassifier(model_path=model_path)
    print("Model loaded successfully!")
    
    # Get sample images
    BASE_DIR = "/home/aorus/workspaces/magiv2"
    speaking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "speaking")
    thinking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "thinking")
    
    samples = []
    
    if os.path.exists(speaking_dir):
        speaking_images = glob.glob(os.path.join(speaking_dir, "*.jpg"))
        samples.extend([(img, "speaking") for img in random.sample(speaking_images, min(num_samples//2, len(speaking_images)))])
    
    if os.path.exists(thinking_dir):
        thinking_images = glob.glob(os.path.join(thinking_dir, "*.jpg"))
        samples.extend([(img, "thinking") for img in random.sample(thinking_images, min(num_samples//2, len(thinking_images)))])
    
    if not samples:
        print("No sample images found!")
        return
    
    # Shuffle samples
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    print(f"\nDemo predictions on {len(samples)} random samples:")
    print("-" * 80)
    
    for i, (img_path, true_label) in enumerate(samples):
        if i >= len(axes):
            break
            
        # Predict
        image = Image.open(img_path).convert("RGB")
        image_array = np.array(image)
        pred_class, confidence = inference.predict_single(image_array)
        
        # Load and display image
        image = Image.open(img_path)
        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Set title with prediction results
        color = 'green' if pred_class == true_label else 'red'
        title = f"True: {true_label}\nPred: {pred_class}\nConf: {confidence:.3f}"
        axes[i].set_title(title, color=color, fontsize=10)
        
        print(f"{os.path.basename(img_path):<30} | True: {true_label:<8} | Pred: {pred_class:<8} | Conf: {confidence:.4f}")
    
    # Hide unused axes
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nDemo visualization saved as 'demo_predictions.png'")

def test_model_performance(model_path: str):
    """Test model performance on validation data"""
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Testing model performance...")
    inference = TailClassifier(model_path=model_path)
    
    BASE_DIR = "/home/aorus/workspaces/magiv2"
    speaking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "speaking")
    thinking_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles", "thinking")
    
    results = {'speaking': {'correct': 0, 'total': 0}, 'thinking': {'correct': 0, 'total': 0}}
    
    # Test speaking images
    if os.path.exists(speaking_dir):
        speaking_images = glob.glob(os.path.join(speaking_dir, "*.jpg"))
        for img_path in speaking_images:
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            pred_class, confidence = inference.predict_single(image_array)
            results['speaking']['total'] += 1
            if pred_class == 'speaking':
                results['speaking']['correct'] += 1
    
    # Test thinking images
    if os.path.exists(thinking_dir):
        thinking_images = glob.glob(os.path.join(thinking_dir, "*.jpg"))
        for img_path in thinking_images:
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            pred_class, confidence = inference.predict_single(image_array)
            results['thinking']['total'] += 1
            if pred_class == 'thinking':
                results['thinking']['correct'] += 1
    
    # Calculate and display results
    print("\nModel Performance:")
    print("-" * 50)
    
    total_correct = 0
    total_samples = 0
    
    for class_name, stats in results.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{class_name.capitalize()}: {stats['correct']}/{stats['total']} ({accuracy:.3f})")
        total_correct += stats['correct']
        total_samples += stats['total']
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.3f})")

def main():
    model_path = os.path.join(BASE_DIR, 'module', 'tail_classification', 'best_tail_classifier.pth')
    
    print("=== Tail Classification Demo ===")
    print("1. Running demo inference...")
    demo_inference(model_path, num_samples=10)
    
    print("\n2. Testing model performance...")
    test_model_performance(model_path)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
