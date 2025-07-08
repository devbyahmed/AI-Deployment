"""
Density Estimation Model
Neural network for crowd density estimation and mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image
import logging

class CSRNet(nn.Module):
    """CSRNet implementation for crowd density estimation"""
    
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        # VGG-16 Frontend (first 10 conv layers)
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Dilated convolutions backend
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        if not load_weights:
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

class MultiScaleDensityNet(nn.Module):
    """Multi-scale density estimation network"""
    
    def __init__(self):
        super(MultiScaleDensityNet, self).__init__()
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
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
        )
        
        # Multi-scale branches
        self.scale1 = self._make_scale_branch(256, [1, 1, 1])
        self.scale2 = self._make_scale_branch(256, [2, 2, 2])
        self.scale3 = self._make_scale_branch(256, [3, 3, 3])
        self.scale4 = self._make_scale_branch(256, [4, 4, 4])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        
        self._initialize_weights()
    
    def _make_scale_branch(self, in_channels, dilations):
        layers = []
        for dilation in dilations:
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.shared_features(x)
        
        # Process through different scales
        scale1_out = self.scale1(features)
        scale2_out = self.scale2(features)
        scale3_out = self.scale3(features)
        scale4_out = self.scale4(features)
        
        # Concatenate and fuse
        multi_scale = torch.cat([scale1_out, scale2_out, scale3_out, scale4_out], dim=1)
        density_map = self.fusion(multi_scale)
        
        return density_map
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class AttentionDensityNet(nn.Module):
    """Density estimation with attention mechanism"""
    
    def __init__(self):
        super(AttentionDensityNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
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
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Attention modules
        self.spatial_attention = SpatialAttentionModule(512)
        self.channel_attention = ChannelAttentionModule(512)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, 1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Apply attention
        features = self.channel_attention(features)
        features = self.spatial_attention(features)
        
        # Decode
        density_map = self.decoder(features)
        
        return density_map
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SpatialAttentionModule(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        return x * attention

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        avg_attention = self.fc(avg_pool)
        max_attention = self.fc(max_pool)
        
        attention = self.sigmoid(avg_attention + max_attention).view(b, c, 1, 1)
        return x * attention

class DensityEstimator:
    """High-level density estimation interface"""
    
    def __init__(self, model_type='csrnet', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_type = model_type
        
        # Initialize model
        if model_type == 'csrnet':
            self.model = CSRNet().to(device)
        elif model_type == 'multiscale':
            self.model = MultiScaleDensityNet().to(device)
        elif model_type == 'attention':
            self.model = AttentionDensityNet().to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()
        
        # Preprocessing
        self.normalize = torch.nn.Sequential(
            torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    def load_weights(self, weight_path):
        """Load pre-trained weights"""
        try:
            checkpoint = torch.load(weight_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logging.info(f"Loaded weights from {weight_path}")
        except Exception as e:
            logging.error(f"Error loading weights: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for density estimation"""
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Convert to tensor
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure proper shape and type
        if len(image.shape) == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            raise ValueError("Image must be 3-dimensional")
        
        # Normalize
        image = self.normalize(image)
        
        return image.unsqueeze(0).to(self.device)
    
    def estimate_density(self, image, return_count=True):
        """Estimate crowd density from image"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Predict
            density_map = self.model(input_tensor)
            
            # Post-process
            density_map = density_map.squeeze().cpu().numpy()
            
            # Apply Gaussian filter for smoothing
            density_map = ndimage.gaussian_filter(density_map, sigma=1)
            
            result = {
                'density_map': density_map,
                'max_density': float(np.max(density_map)),
                'mean_density': float(np.mean(density_map)),
                'total_density': float(np.sum(density_map))
            }
            
            if return_count:
                # Estimate count (sum of density map)
                count = np.sum(density_map)
                result['estimated_count'] = max(0, int(round(count)))
            
            return result
    
    def generate_heatmap(self, image, density_map=None, colormap=cv2.COLORMAP_JET):
        """Generate density heatmap visualization"""
        if density_map is None:
            result = self.estimate_density(image)
            density_map = result['density_map']
        
        # Normalize density map to 0-255
        normalized_density = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(normalized_density, colormap)
        
        # Resize to match original image size
        if isinstance(image, np.ndarray):
            target_height, target_width = image.shape[:2]
        else:
            target_width, target_height = image.size
        
        heatmap = cv2.resize(heatmap, (target_width, target_height))
        
        return heatmap
    
    def overlay_heatmap(self, image, density_map=None, alpha=0.6):
        """Overlay density heatmap on original image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Generate heatmap
        heatmap = self.generate_heatmap(image, density_map)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    def analyze_density_regions(self, density_map, threshold_percentile=75):
        """Analyze high-density regions in the density map"""
        # Calculate threshold
        threshold = np.percentile(density_map, threshold_percentile)
        
        # Find high-density regions
        binary_map = (density_map > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)
        
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            centroid = centroids[i]
            bbox = [
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            ]
            
            # Calculate region density
            region_mask = (labels == i)
            region_density = np.sum(density_map[region_mask])
            
            regions.append({
                'id': i,
                'centroid': centroid.tolist(),
                'bbox': bbox,
                'area': int(area),
                'density': float(region_density),
                'avg_density': float(region_density / area) if area > 0 else 0
            })
        
        # Sort by density
        regions.sort(key=lambda x: x['density'], reverse=True)
        
        return regions
    
    def batch_estimate(self, images):
        """Batch estimation for multiple images"""
        results = []
        
        for image in images:
            try:
                result = self.estimate_density(image)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing image: {e}")
                results.append({
                    'density_map': np.zeros((10, 10)),
                    'estimated_count': 0,
                    'max_density': 0,
                    'mean_density': 0,
                    'total_density': 0,
                    'error': str(e)
                })
        
        return results
    
    def temporal_density_analysis(self, image_sequence, window_size=5):
        """Analyze density changes over time"""
        if len(image_sequence) < 2:
            return {}
        
        # Estimate density for each frame
        density_estimates = []
        for image in image_sequence:
            result = self.estimate_density(image)
            density_estimates.append(result['estimated_count'])
        
        # Calculate temporal metrics
        density_changes = np.diff(density_estimates)
        
        # Moving average
        if len(density_estimates) >= window_size:
            moving_avg = np.convolve(density_estimates, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        else:
            moving_avg = density_estimates
        
        return {
            'density_sequence': density_estimates,
            'density_changes': density_changes.tolist(),
            'moving_average': moving_avg.tolist(),
            'trend': 'increasing' if density_changes[-3:].mean() > 0 else 'decreasing',
            'volatility': float(np.std(density_changes)),
            'max_density': float(np.max(density_estimates)),
            'min_density': float(np.min(density_estimates)),
            'avg_density': float(np.mean(density_estimates))
        }