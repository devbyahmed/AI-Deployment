"""
Crowd Detection Module
Custom crowd detection using PyTorch and computer vision techniques
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import time
from collections import deque
import threading
import logging

class CrowdDetectionModel(nn.Module):
    """Custom PyTorch model for crowd detection"""
    
    def __init__(self, num_classes=2):  # person/no-person
        super(CrowdDetectionModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class YOLODetector:
    """YOLO-based person detection for crowd analysis"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_yolo_model(model_path)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def load_yolo_model(self, model_path):
        """Load YOLO model"""
        try:
            # Try to load custom model if provided
            if model_path:
                model = torch.jit.load(model_path)
            else:
                # Use torchvision's pre-trained model as fallback
                import torchvision.models as models
                model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            # Return a simple detector as fallback
            return self.create_simple_detector()
    
    def create_simple_detector(self):
        """Create a simple HOG-based person detector as fallback"""
        return cv2.HOGDescriptor_getDefaultPeopleDetector()
    
    def detect_persons(self, frame):
        """Detect persons in the frame"""
        try:
            # Preprocess frame
            if isinstance(frame, np.ndarray):
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
            else:
                image = frame
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Process predictions
            persons = self.process_predictions(predictions, frame.shape)
            return persons
            
        except Exception as e:
            logging.error(f"Error in person detection: {e}")
            # Fallback to HOG detector
            return self.hog_detect_persons(frame)
    
    def hog_detect_persons(self, frame):
        """Fallback HOG-based person detection"""
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people
            boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
            
            persons = []
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight > 0.5:  # Confidence threshold
                    persons.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': float(weight),
                        'class': 'person'
                    })
            
            return persons
        except Exception as e:
            logging.error(f"Error in HOG detection: {e}")
            return []
    
    def process_predictions(self, predictions, frame_shape):
        """Process model predictions and return person detections"""
        persons = []
        
        if len(predictions) > 0:
            pred = predictions[0]
            
            # Extract boxes, scores, and labels
            if 'boxes' in pred:
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                # Filter for person class and confidence threshold
                for box, score, label in zip(boxes, scores, labels):
                    if label == self.person_class_id + 1 and score > self.confidence_threshold:  # +1 for COCO indexing
                        x1, y1, x2, y2 = box
                        persons.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(score),
                            'class': 'person'
                        })
        
        return persons

class CrowdAnalyzer:
    """Main crowd analysis class that combines detection with analytics"""
    
    def __init__(self, model_path=None):
        self.detector = YOLODetector(model_path)
        self.frame_buffer = deque(maxlen=30)  # Store last 30 frames for analysis
        self.person_tracks = {}  # Track persons across frames
        self.next_track_id = 0
        self.tracking_threshold = 50  # pixels
        
        # Analytics
        self.density_calculator = DensityCalculator()
        self.movement_analyzer = MovementAnalyzer()
        
    def analyze_frame(self, frame, stream_id=0):
        """Analyze a single frame for crowd metrics"""
        start_time = time.time()
        
        # Detect persons
        persons = self.detector.detect_persons(frame)
        
        # Update tracking
        tracked_persons = self.update_tracking(persons, stream_id)
        
        # Calculate metrics
        metrics = {
            'timestamp': time.time(),
            'stream_id': stream_id,
            'person_count': len(persons),
            'persons': tracked_persons,
            'density': self.density_calculator.calculate_density(persons, frame.shape),
            'movement': self.movement_analyzer.analyze_movement(tracked_persons),
            'processing_time': time.time() - start_time
        }
        
        # Store frame for temporal analysis
        self.frame_buffer.append({
            'frame': frame,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        return metrics
    
    def update_tracking(self, current_persons, stream_id):
        """Update person tracking across frames"""
        tracked_persons = []
        
        # Simple centroid-based tracking
        for person in current_persons:
            bbox = person['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            # Find closest existing track
            best_match = None
            min_distance = float('inf')
            
            for track_id, track in self.person_tracks.items():
                if track['stream_id'] == stream_id:
                    distance = np.sqrt((center[0] - track['center'][0])**2 + 
                                     (center[1] - track['center'][1])**2)
                    if distance < min_distance and distance < self.tracking_threshold:
                        min_distance = distance
                        best_match = track_id
            
            if best_match:
                # Update existing track
                self.person_tracks[best_match].update({
                    'center': center,
                    'bbox': bbox,
                    'confidence': person['confidence'],
                    'last_seen': time.time()
                })
                person['track_id'] = best_match
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.person_tracks[track_id] = {
                    'track_id': track_id,
                    'stream_id': stream_id,
                    'center': center,
                    'bbox': bbox,
                    'confidence': person['confidence'],
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'trajectory': [center]
                }
                person['track_id'] = track_id
            
            tracked_persons.append(person)
        
        # Remove old tracks
        current_time = time.time()
        old_tracks = [tid for tid, track in self.person_tracks.items() 
                     if current_time - track['last_seen'] > 5.0]  # 5 seconds timeout
        for tid in old_tracks:
            del self.person_tracks[tid]
        
        return tracked_persons
    
    def get_crowd_statistics(self, stream_id=None):
        """Get comprehensive crowd statistics"""
        if stream_id is not None:
            tracks = [t for t in self.person_tracks.values() if t['stream_id'] == stream_id]
        else:
            tracks = list(self.person_tracks.values())
        
        if not tracks:
            return {
                'total_count': 0,
                'active_tracks': 0,
                'avg_confidence': 0,
                'density_level': 'low'
            }
        
        total_count = len(tracks)
        avg_confidence = np.mean([t['confidence'] for t in tracks])
        
        # Determine density level
        if total_count > 50:
            density_level = 'very_high'
        elif total_count > 30:
            density_level = 'high'
        elif total_count > 15:
            density_level = 'medium'
        elif total_count > 5:
            density_level = 'low'
        else:
            density_level = 'very_low'
        
        return {
            'total_count': total_count,
            'active_tracks': len([t for t in tracks if time.time() - t['last_seen'] < 1.0]),
            'avg_confidence': float(avg_confidence),
            'density_level': density_level,
            'tracks': tracks
        }

class DensityCalculator:
    """Calculate crowd density metrics"""
    
    def __init__(self):
        self.grid_size = (10, 10)  # 10x10 grid for density calculation
    
    def calculate_density(self, persons, frame_shape):
        """Calculate crowd density using grid-based approach"""
        if not persons:
            return {
                'overall_density': 0,
                'density_map': np.zeros(self.grid_size),
                'hotspots': []
            }
        
        height, width = frame_shape[:2]
        grid_h, grid_w = self.grid_size
        
        # Create density map
        density_map = np.zeros(self.grid_size)
        
        for person in persons:
            bbox = person['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Map to grid coordinates
            grid_x = min(int(center_x / width * grid_w), grid_w - 1)
            grid_y = min(int(center_y / height * grid_h), grid_h - 1)
            
            density_map[grid_y, grid_x] += 1
        
        # Calculate overall density (persons per unit area)
        total_area = width * height
        overall_density = len(persons) / total_area * 10000  # persons per 10,000 pixels
        
        # Find hotspots (grid cells with high density)
        hotspots = []
        threshold = np.mean(density_map) + 2 * np.std(density_map)
        
        for i in range(grid_h):
            for j in range(grid_w):
                if density_map[i, j] > threshold:
                    hotspots.append({
                        'grid_position': (i, j),
                        'density': float(density_map[i, j]),
                        'coordinates': (
                            int(j * width / grid_w),
                            int(i * height / grid_h),
                            int((j + 1) * width / grid_w),
                            int((i + 1) * height / grid_h)
                        )
                    })
        
        return {
            'overall_density': float(overall_density),
            'density_map': density_map.tolist(),
            'hotspots': hotspots,
            'max_density': float(np.max(density_map)),
            'avg_density': float(np.mean(density_map))
        }

class MovementAnalyzer:
    """Analyze crowd movement patterns"""
    
    def __init__(self):
        self.velocity_threshold = 5.0  # pixels per frame
        self.direction_bins = 8  # 8 directional bins (45 degrees each)
    
    def analyze_movement(self, tracked_persons):
        """Analyze movement patterns of tracked persons"""
        if not tracked_persons:
            return {
                'avg_speed': 0,
                'movement_directions': [0] * self.direction_bins,
                'stationary_count': 0,
                'moving_count': 0
            }
        
        speeds = []
        directions = []
        stationary_count = 0
        moving_count = 0
        
        for person in tracked_persons:
            if 'track_id' in person:
                # Calculate speed and direction from trajectory
                track_id = person['track_id']
                # This would use trajectory data from tracking
                # For now, we'll use a simplified calculation
                
                # Simulate movement analysis
                speed = np.random.uniform(0, 10)  # Placeholder
                direction = np.random.uniform(0, 2 * np.pi)  # Placeholder
                
                speeds.append(speed)
                directions.append(direction)
                
                if speed < self.velocity_threshold:
                    stationary_count += 1
                else:
                    moving_count += 1
        
        # Calculate direction histogram
        direction_histogram = [0] * self.direction_bins
        for direction in directions:
            bin_index = int((direction / (2 * np.pi)) * self.direction_bins) % self.direction_bins
            direction_histogram[bin_index] += 1
        
        return {
            'avg_speed': float(np.mean(speeds)) if speeds else 0,
            'max_speed': float(np.max(speeds)) if speeds else 0,
            'movement_directions': direction_histogram,
            'stationary_count': stationary_count,
            'moving_count': moving_count,
            'movement_ratio': moving_count / len(tracked_persons) if tracked_persons else 0
        }