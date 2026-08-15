import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import glob
import timm
import argparse
from typing import List, Tuple

from settings import DEVICE

class TailClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=False, model_path=None):
        super(TailClassifier, self).__init__()
        self.device = DEVICE
        
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # Remove classifier
        feature_dim = self.backbone.num_features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        # Transform for inference
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.ToGray(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.class_names = ['speaking', 'thinking']
        
        if model_path:
            self.load_model(model_path)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def load_model(self, model_path):
        # Load the entire model state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()
        
    def predict_single(self, image: np.array) -> Tuple[str, float]:
        """Predict single image"""
        # Load and preprocess image
        image = self.transform(image=image)["image"]
        image = image.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self(image)  # Use the full model forward pass
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        class_name = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return class_name, confidence_score
    
    def predict_batch(self, images: List[np.array]) -> List[Tuple[str, float]]:
        """Predict batch of images"""
        results = []
        for image in images:
            try:
                class_name, confidence = self.predict_single(image)
                results.append((class_name, confidence))
            except Exception as e:
                print(f"Error processing {image}: {e}")
                results.append(('error', 0.0))
        return results


def main():
    parser = argparse.ArgumentParser(description='Tail Classification Inference')
    parser.add_argument('--input', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    
    args = parser.parse_args()
    
    # Initialize inference
    print(f"Loading model from {args.model_path}")
    inference = TailClassifier(model_path=args.model_path)
    print(f"Model loaded successfully on {inference.device}")
    
    # Check if input is file or folder
    if os.path.isfile(args.input):
        # Single image
        print(f"Predicting single image: {args.input}")
        image = np.array(Image.open(args.input).convert("RGB"))
        class_name, confidence = inference.predict_single(image)
        print(f"Prediction: {class_name} (confidence: {confidence:.4f})")
        
    else:
        print(f"Error: {args.input} is not a valid file")

if __name__ == "__main__":
    main()
