"""
Crowd Detection Model
Custom trained model for crowd detection and person counting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import os
import logging
from typing import Tuple, List, Dict, Optional

class CrowdDetectionCNN(nn.Module):
    """Custom CNN for crowd detection and counting"""
    
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super(CrowdDetectionCNN, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            backbone_features = self.backbone.classifier[6].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            backbone_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Crowd-specific layers
        self.crowd_features = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Multiple output heads
        self.count_head = nn.Linear(512, 1)  # Person count
        self.density_head = nn.Linear(512, 1)  # Density estimation
        self.classification_head = nn.Linear(512, num_classes)  # Crowd classification
        
        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention(backbone_features)
        
    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(features)
        
        # Process through crowd-specific layers
        crowd_features = self.crowd_features(attended_features)
        
        # Multiple outputs
        count = self.count_head(crowd_features)
        density = self.density_head(crowd_features)
        classification = self.classification_head(crowd_features)
        
        return {
            'count': count,
            'density': density,
            'classification': classification,
            'features': crowd_features
        }

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on crowd regions"""
    
    def __init__(self, in_features):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class DensityEstimationModel(nn.Module):
    """Specialized model for crowd density estimation using CSRNet architecture"""
    
    def __init__(self):
        super(DensityEstimationModel, self).__init__()
        
        # Frontend: VGG-16 first 10 layers
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Backend: Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CrowdDataset(Dataset):
    """Dataset class for crowd detection training"""
    
    def __init__(self, image_paths, labels=None, transform=None, mode='train'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

class CrowdModelTrainer:
    """Trainer class for crowd detection models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4):
        """Setup training components"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.1
        )
        
        # Multi-task loss
        self.criterion = {
            'count': nn.MSELoss(),
            'density': nn.MSELoss(),
            'classification': nn.CrossEntropyLoss()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            
            # Handle different target formats
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.calculate_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logging.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def calculate_loss(self, outputs, targets):
        """Calculate multi-task loss"""
        total_loss = 0
        
        if isinstance(targets, dict):
            # Multi-task targets
            if 'count' in targets and 'count' in outputs:
                count_loss = self.criterion['count'](outputs['count'], targets['count'])
                total_loss += count_loss
            
            if 'density' in targets and 'density' in outputs:
                density_loss = self.criterion['density'](outputs['density'], targets['density'])
                total_loss += density_loss
            
            if 'classification' in targets and 'classification' in outputs:
                class_loss = self.criterion['classification'](outputs['classification'], targets['classification'])
                total_loss += class_loss
        else:
            # Single task (count prediction)
            if 'count' in outputs:
                total_loss = self.criterion['count'](outputs['count'].squeeze(), targets.float())
            else:
                total_loss = self.criterion['count'](outputs.squeeze(), targets.float())
        
        return total_loss
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.calculate_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # Store predictions for metrics
                if isinstance(outputs, dict) and 'count' in outputs:
                    predictions.extend(outputs['count'].cpu().numpy())
                else:
                    predictions.extend(outputs.cpu().numpy())
                
                if isinstance(targets, dict) and 'count' in targets:
                    targets_list.extend(targets['count'].cpu().numpy())
                else:
                    targets_list.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets_list = np.array(targets_list)
        
        mae = np.mean(np.abs(predictions - targets_list))
        mse = np.mean((predictions - targets_list) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    
    def save_model(self, filepath, epoch, best_loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': best_loss,
        }, filepath)
        
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))

class CrowdModelInference:
    """Inference class for crowd detection models"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.transform = self._get_transform()
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found: {model_path}, using pre-trained model")
            self.model = CrowdDetectionCNN(pretrained=True).to(self.device)
            self.model.eval()
            return
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine model architecture from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Initialize model based on state dict keys
            if any('density_head' in key for key in state_dict.keys()):
                self.model = CrowdDetectionCNN().to(self.device)
            else:
                self.model = DensityEstimationModel().to(self.device)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logging.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Fallback to pre-trained model
            self.model = CrowdDetectionCNN(pretrained=True).to(self.device)
            self.model.eval()
    
    def _get_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image):
        """Predict crowd metrics for a single image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Process outputs
        if isinstance(outputs, dict):
            result = {
                'count': outputs.get('count', torch.tensor([0])).item(),
                'density': outputs.get('density', torch.tensor([0])).item(),
                'classification': outputs.get('classification', torch.tensor([0])).argmax().item()
            }
        else:
            # For density estimation model
            density_map = outputs.squeeze().cpu().numpy()
            count = np.sum(density_map)
            result = {
                'count': float(count),
                'density': float(np.mean(density_map)),
                'density_map': density_map.tolist()
            }
        
        return result
    
    def batch_predict(self, images):
        """Predict crowd metrics for a batch of images"""
        batch_results = []
        
        for image in images:
            result = self.predict(image)
            batch_results.append(result)
        
        return batch_results

def create_data_transforms():
    """Create data augmentation transforms for training"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def train_crowd_model(train_data, val_data, epochs=50, batch_size=32, learning_rate=0.001):
    """Complete training pipeline for crowd detection model"""
    
    # Setup data transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = CrowdDataset(train_data['images'], train_data['labels'], train_transform, 'train')
    val_dataset = CrowdDataset(val_data['images'], val_data['labels'], val_transform, 'val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model and trainer
    model = CrowdDetectionCNN(num_classes=1, backbone='resnet50', pretrained=True)
    trainer = CrowdModelTrainer(model)
    trainer.setup_training(learning_rate=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        val_loss = val_metrics['loss']
        
        # Learning rate scheduling
        trainer.scheduler.step()
        
        logging.info(
            f'Epoch {epoch}/{epochs}: '
            f'Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}, '
            f'Val MAE: {val_metrics["mae"]:.4f}, '
            f'Val RMSE: {val_metrics["rmse"]:.4f}'
        )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_model(f'best_crowd_model_epoch_{epoch}.pth', epoch, best_loss)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch}')
            break
    
    return trainer.model