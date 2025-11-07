"""
Training Module
PyTorch-based training pipeline for image classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json


class STPImageDataset(Dataset):
    """Dataset for STP rendered images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels (class indices)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class STAPClassifier:
    """Classifier for STP files based on rendered images."""
    
    def __init__(self, 
                 num_classes: int,
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 device: Optional[str] = None):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of STP file classes to classify
            model_name: Name of pre-trained model to use
            pretrained: Whether to use pre-trained weights
            device: Device to use for training (cuda/cpu)
        """
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pre-trained model
        self.model = self._load_model(pretrained)
        self.model = self.model.to(self.device)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = []
        
    def _load_model(self, pretrained: bool) -> nn.Module:
        """
        Load pre-trained model and modify for classification.
        
        Args:
            pretrained: Whether to use pre-trained weights
            
        Returns:
            Modified model
        """
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model
    
    def prepare_data(self, 
                     image_dir: str,
                     class_names: List[str],
                     train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders from rendered images.
        
        Args:
            image_dir: Directory containing rendered images
            class_names: List of class names (STP file names)
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        self.class_names = class_names
        image_dir = Path(image_dir)
        
        # Collect all images and labels
        all_images = []
        all_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            # Find all images for this class
            class_images = list(image_dir.glob(f"{class_name}_*.png"))
            all_images.extend([str(img) for img in class_images])
            all_labels.extend([class_idx] * len(class_images))
        
        # Shuffle and split
        indices = np.random.permutation(len(all_images))
        split_idx = int(len(indices) * train_split)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_images = [all_images[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        
        val_images = [all_images[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        
        # Create datasets
        train_dataset = STPImageDataset(train_images, train_labels, self.train_transform)
        val_dataset = STPImageDataset(val_images, val_labels, self.val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        return train_loader, val_loader
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 0.001,
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Path to save the best model
            
        Returns:
            Dictionary with training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc="Training"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                self.save_model(save_path)
                print(f"Model saved to {save_path}")
        
        return history
    
    def predict(self, image_path: str) -> Tuple[int, float, str]:
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (predicted_class_idx, confidence, class_name)
        """
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_idx = predicted.item()
        confidence_val = confidence.item()
        class_name = self.class_names[predicted_idx] if self.class_names else f"Class_{predicted_idx}"
        
        return predicted_idx, confidence_val, class_name
    
    def save_model(self, path: str):
        """Save model weights and metadata."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'class_names': self.class_names
        }
        torch.save(save_dict, path)
    
    def load_model(self, path: str):
        """Load model weights and metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names', [])
        self.num_classes = checkpoint['num_classes']
