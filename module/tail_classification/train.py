import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

from module.tail_classification.model import TailClassifier
from settings import BASE_DIR
warnings.filterwarnings('ignore')
import timm


# Dataset custom
class TailDataset(Dataset):
    def __init__(self, image_paths,labels, train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.train = train

        # Augmentation cân bằng cho "thinking" (class 1)
        self.augment_thinking = A.Compose([
            A.ToGray(p=1.0),
            A.HorizontalFlip(p=0.3),  # Giảm từ 0.5 xuống 0.3
            A.VerticalFlip(p=0.1),    # Giảm từ 0.2 xuống 0.1
            A.RandomRotate90(p=0.2),  # Giảm từ 0.5 xuống 0.2
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),  # Giảm intensity
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # Giảm intensity
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),  # Giảm noise
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Augmentation nhẹ cho "speaking" (class 0)
        self.augment_speaking = A.Compose([
            A.ToGray(p=1.0),
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Validation augmentation (chỉ resize + normalize)
        self.transform_val = A.Compose([
            A.ToGray(p=1.0),
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.train:
            if label == 1:  # thinking
                image = self.augment_thinking(image=image)["image"]
            else:  # speaking
                image = self.augment_speaking(image=image)["image"]
        else:
            image = self.transform_val(image=image)["image"]

        return image, label

def load_data(data_dir):
    """Load data from cropped_bubbles directory"""
    speaking_dir = os.path.join(data_dir, "speaking")
    thinking_dir = os.path.join(data_dir, "thinking")
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    
    # Speaking images (label 0)
    speaking_paths = glob.glob(os.path.join(speaking_dir, "*.jpg"))
    image_paths.extend(speaking_paths)
    labels.extend([0] * len(speaking_paths))
    
    # Thinking images (label 1)
    thinking_paths = glob.glob(os.path.join(thinking_dir, "*.jpg"))
    image_paths.extend(thinking_paths)
    labels.extend([1] * len(thinking_paths))
    
    return image_paths, labels


def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, device='cuda', model_save_path=None, class_weights=None):
    """Train the model with early stopping"""
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Training on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        print('-' * 50)
    
    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                 target_names=['speaking', 'thinking'])
    
    return accuracy, report, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_classification_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['speaking', 'thinking'],
                yticklabels=['speaking', 'thinking'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('training_classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Configuration
    BASE_DIR = "/home/aorus/workspaces/magiv2"
    DATA_DIR = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles")
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    image_paths, labels = load_data(DATA_DIR)
    print(f"Total images: {len(image_paths)}")
    print(f"Speaking: {labels.count(0)}, Thinking: {labels.count(1)}")
    
    # Calculate class weights for imbalanced dataset
    from collections import Counter
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = []
    for i in range(2):
        weight = total_samples / (2.0 * class_counts[i])
        class_weights.append(weight)
    
    print(f"Class weights: {class_weights}")
    print(f"Speaking weight: {class_weights[0]:.3f}, Thinking weight: {class_weights[1]:.3f}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=TEST_SIZE + VAL_SIZE, 
        random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VAL_SIZE), 
        random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
     # Create model
    print("Creating model...")
    model = TailClassifier(model_name='efficientnet_b0', pretrained=True)
    model = model.to(device)   
    
    # Create datasets
    train_dataset = TailDataset(X_train, y_train, train=True)
    val_dataset = TailDataset(X_val, y_val, train=False)
    test_dataset = TailDataset(X_test, y_test, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    

    
    # Train model
    print("Starting training...")
    model_save_path = os.path.join(BASE_DIR, 'module', 'tail_classification', 'best_tail_classifier.pth')
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device, 
        model_save_path=model_save_path, class_weights=class_weights
    )
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_accuracy, test_report, test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(test_report)
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(test_labels, test_preds)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
