"""
Movement Tracker Module
Advanced object tracking and movement analysis for crowd intelligence
"""

import numpy as np
import cv2
from collections import defaultdict, deque
import time
import math
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import logging

class MovementTracker:
    """Advanced multi-object tracker for crowd movement analysis"""
    
    def __init__(self, max_disappeared=10, max_distance=50):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        
        # Movement analysis
        self.velocity_history = defaultdict(deque)
        self.trajectory_history = defaultdict(deque)
        self.direction_history = defaultdict(deque)
        
        # Analytics
        self.movement_patterns = {}
        self.flow_vectors = []
        
    def register(self, centroid, bbox, confidence=1.0):
        """Register a new object for tracking"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'confidence': confidence,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'frames_tracked': 1
        }
        
        # Initialize movement history
        self.velocity_history[self.next_object_id] = deque(maxlen=10)
        self.trajectory_history[self.next_object_id] = deque(maxlen=50)
        self.direction_history[self.next_object_id] = deque(maxlen=10)
        
        # Add initial trajectory point
        self.trajectory_history[self.next_object_id].append({
            'position': centroid,
            'timestamp': time.time()
        })
        
        self.next_object_id += 1
        
        return self.next_object_id - 1
    
    def deregister(self, object_id):
        """Deregister an object from tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
            del self.velocity_history[object_id]
            del self.trajectory_history[object_id]
            del self.direction_history[object_id]
            
            if object_id in self.disappeared:
                del self.disappeared[object_id]
    
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.get_tracking_results()
        
        # Initialize centroids array
        input_centroids = []
        input_bboxes = []
        input_confidences = []
        
        for detection in detections:
            bbox = detection['bbox']
            centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            input_centroids.append(centroid)
            input_bboxes.append(bbox)
            input_confidences.append(detection.get('confidence', 1.0))
        
        if len(self.objects) == 0:
            # No existing objects, register all detections
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i], input_confidences[i])
        else:
            # Match existing objects with new detections
            self.match_objects(input_centroids, input_bboxes, input_confidences)
        
        return self.get_tracking_results()
    
    def match_objects(self, input_centroids, input_bboxes, input_confidences):
        """Match existing objects with new detections using Hungarian algorithm"""
        object_centroids = [obj['centroid'] for obj in self.objects.values()]
        object_ids = list(self.objects.keys())
        
        if len(object_centroids) > 0 and len(input_centroids) > 0:
            # Compute distance matrix
            D = distance.cdist(np.array(object_centroids), np.array(input_centroids))
            
            # Find optimal assignment
            if D.shape[0] <= D.shape[1]:
                row_indices, col_indices = linear_sum_assignment(D)
            else:
                col_indices, row_indices = linear_sum_assignment(D.T)
            
            # Track used indices
            used_row_indices = set()
            used_col_indices = set()
            
            # Update matched objects
            for (row, col) in zip(row_indices, col_indices):
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.update_object(object_id, input_centroids[col], 
                                     input_bboxes[col], input_confidences[col])
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # Mark unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Register new objects for unmatched detections
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col], 
                                input_confidences[col])
        
        # Register all detections if no existing objects
        elif len(input_centroids) > 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i], input_confidences[i])
    
    def update_object(self, object_id, centroid, bbox, confidence):
        """Update an existing object with new detection"""
        old_centroid = self.objects[object_id]['centroid']
        current_time = time.time()
        
        # Update object properties
        self.objects[object_id]['centroid'] = centroid
        self.objects[object_id]['bbox'] = bbox
        self.objects[object_id]['confidence'] = confidence
        self.objects[object_id]['last_seen'] = current_time
        self.objects[object_id]['frames_tracked'] += 1
        
        # Reset disappeared counter
        self.disappeared[object_id] = 0
        
        # Calculate movement metrics
        self.calculate_movement_metrics(object_id, old_centroid, centroid, current_time)
        
        # Update trajectory
        self.trajectory_history[object_id].append({
            'position': centroid,
            'timestamp': current_time
        })
    
    def calculate_movement_metrics(self, object_id, old_centroid, new_centroid, timestamp):
        """Calculate velocity, direction, and other movement metrics"""
        # Calculate displacement
        dx = new_centroid[0] - old_centroid[0]
        dy = new_centroid[1] - old_centroid[1]
        displacement = math.sqrt(dx**2 + dy**2)
        
        # Calculate time difference (assume fixed frame rate if not available)
        last_trajectory = self.trajectory_history[object_id]
        if len(last_trajectory) > 0:
            time_diff = timestamp - last_trajectory[-1]['timestamp']
        else:
            time_diff = 1/30  # Assume 30 FPS
        
        # Calculate velocity (pixels per second)
        velocity = displacement / time_diff if time_diff > 0 else 0
        self.velocity_history[object_id].append(velocity)
        
        # Calculate direction (radians)
        if displacement > 1:  # Only calculate direction if there's significant movement
            direction = math.atan2(dy, dx)
            self.direction_history[object_id].append(direction)
        
        # Update flow vectors for overall crowd flow analysis
        if displacement > 2:  # Threshold for significant movement
            self.flow_vectors.append({
                'start': old_centroid,
                'end': new_centroid,
                'magnitude': displacement,
                'direction': math.atan2(dy, dx),
                'timestamp': timestamp
            })
            
            # Keep only recent flow vectors
            cutoff_time = timestamp - 5.0  # 5 seconds
            self.flow_vectors = [fv for fv in self.flow_vectors if fv['timestamp'] > cutoff_time]
    
    def get_tracking_results(self):
        """Get current tracking results with movement analysis"""
        results = []
        
        for object_id, obj in self.objects.items():
            # Calculate movement statistics
            velocities = list(self.velocity_history[object_id])
            directions = list(self.direction_history[object_id])
            trajectory = list(self.trajectory_history[object_id])
            
            avg_velocity = np.mean(velocities) if velocities else 0
            max_velocity = np.max(velocities) if velocities else 0
            
            # Calculate dominant direction
            if directions:
                # Convert to unit vectors and average
                x_components = [math.cos(d) for d in directions]
                y_components = [math.sin(d) for d in directions]
                avg_direction = math.atan2(np.mean(y_components), np.mean(x_components))
            else:
                avg_direction = 0
            
            # Determine movement state
            movement_state = self.classify_movement_state(velocities, directions)
            
            result = {
                'track_id': object_id,
                'centroid': obj['centroid'],
                'bbox': obj['bbox'],
                'confidence': obj['confidence'],
                'frames_tracked': obj['frames_tracked'],
                'first_seen': obj['first_seen'],
                'last_seen': obj['last_seen'],
                'movement': {
                    'avg_velocity': avg_velocity,
                    'max_velocity': max_velocity,
                    'current_velocity': velocities[-1] if velocities else 0,
                    'avg_direction': avg_direction,
                    'movement_state': movement_state,
                    'trajectory_length': len(trajectory),
                    'total_distance': self.calculate_total_distance(trajectory)
                }
            }
            
            results.append(result)
        
        return results
    
    def classify_movement_state(self, velocities, directions):
        """Classify the movement state of an object"""
        if not velocities:
            return 'unknown'
        
        avg_velocity = np.mean(velocities)
        velocity_std = np.std(velocities) if len(velocities) > 1 else 0
        
        # Thresholds for movement classification
        stationary_threshold = 2.0
        slow_threshold = 5.0
        moderate_threshold = 15.0
        
        if avg_velocity < stationary_threshold:
            return 'stationary'
        elif avg_velocity < slow_threshold:
            if velocity_std > avg_velocity * 0.5:
                return 'irregular'
            else:
                return 'slow'
        elif avg_velocity < moderate_threshold:
            return 'moderate'
        else:
            return 'fast'
    
    def calculate_total_distance(self, trajectory):
        """Calculate total distance traveled in trajectory"""
        if len(trajectory) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]['position']
            p2 = trajectory[i]['position']
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += distance
        
        return total_distance
    
    def get_crowd_flow_analysis(self):
        """Analyze overall crowd flow patterns"""
        if not self.flow_vectors:
            return {
                'dominant_direction': 0,
                'flow_magnitude': 0,
                'flow_consistency': 0,
                'congestion_level': 'low'
            }
        
        # Calculate dominant flow direction
        directions = [fv['direction'] for fv in self.flow_vectors]
        magnitudes = [fv['magnitude'] for fv in self.flow_vectors]
        
        # Weight directions by magnitude
        x_components = [mag * math.cos(dir) for mag, dir in zip(magnitudes, directions)]
        y_components = [mag * math.sin(dir) for mag, dir in zip(magnitudes, directions)]
        
        dominant_direction = math.atan2(np.mean(y_components), np.mean(x_components))
        flow_magnitude = math.sqrt(np.mean(x_components)**2 + np.mean(y_components)**2)
        
        # Calculate flow consistency (how aligned the movements are)
        direction_variance = np.var(directions)
        flow_consistency = 1.0 / (1.0 + direction_variance)  # High consistency = low variance
        
        # Determine congestion level based on movement patterns
        avg_magnitude = np.mean(magnitudes)
        if avg_magnitude < 2:
            congestion_level = 'high'
        elif avg_magnitude < 5:
            congestion_level = 'medium'
        else:
            congestion_level = 'low'
        
        return {
            'dominant_direction': dominant_direction,
            'flow_magnitude': flow_magnitude,
            'flow_consistency': flow_consistency,
            'congestion_level': congestion_level,
            'active_objects': len(self.objects),
            'total_flow_vectors': len(self.flow_vectors)
        }
    
    def get_movement_heatmap(self, frame_shape, grid_size=(20, 20)):
        """Generate movement heatmap for visualization"""
        height, width = frame_shape[:2]
        grid_h, grid_w = grid_size
        
        # Initialize heatmap
        movement_heatmap = np.zeros(grid_size)
        direction_heatmap = np.zeros(grid_size)
        
        for fv in self.flow_vectors:
            # Map flow vector to grid
            start_x, start_y = fv['start']
            grid_x = min(int(start_x / width * grid_w), grid_w - 1)
            grid_y = min(int(start_y / height * grid_h), grid_h - 1)
            
            # Add magnitude to movement heatmap
            movement_heatmap[grid_y, grid_x] += fv['magnitude']
            
            # Add direction information
            direction_heatmap[grid_y, grid_x] = fv['direction']
        
        return {
            'movement_intensity': movement_heatmap.tolist(),
            'movement_directions': direction_heatmap.tolist(),
            'grid_size': grid_size
        }

class OpticalFlowAnalyzer:
    """Optical flow analysis for dense motion estimation"""
    
    def __init__(self):
        self.prev_frame = None
        self.flow_history = deque(maxlen=10)
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
    
    def analyze_flow(self, frame):
        """Analyze optical flow in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return {
                'flow_magnitude': 0,
                'flow_direction': 0,
                'dense_flow': None
            }
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, None, None, **self.lk_params
        )
        
        if flow[0] is not None:
            # Calculate flow statistics
            flow_vectors = flow[0] - flow[1] if flow[1] is not None else flow[0]
            
            # Remove invalid vectors
            valid_flow = flow_vectors[flow[2].ravel() == 1] if flow[2] is not None else flow_vectors
            
            if len(valid_flow) > 0:
                magnitudes = np.sqrt(valid_flow[:, 0]**2 + valid_flow[:, 1]**2)
                directions = np.arctan2(valid_flow[:, 1], valid_flow[:, 0])
                
                avg_magnitude = np.mean(magnitudes)
                avg_direction = np.mean(directions)
                
                flow_data = {
                    'flow_magnitude': float(avg_magnitude),
                    'flow_direction': float(avg_direction),
                    'dense_flow': valid_flow.tolist(),
                    'num_vectors': len(valid_flow)
                }
            else:
                flow_data = {
                    'flow_magnitude': 0,
                    'flow_direction': 0,
                    'dense_flow': None,
                    'num_vectors': 0
                }
        else:
            flow_data = {
                'flow_magnitude': 0,
                'flow_direction': 0,
                'dense_flow': None,
                'num_vectors': 0
            }
        
        self.flow_history.append(flow_data)
        self.prev_frame = gray
        
        return flow_data
    
    def get_flow_trends(self):
        """Get temporal trends in optical flow"""
        if len(self.flow_history) < 2:
            return {
                'magnitude_trend': 'stable',
                'direction_trend': 'stable',
                'flow_acceleration': 0
            }
        
        magnitudes = [fh['flow_magnitude'] for fh in self.flow_history]
        directions = [fh['flow_direction'] for fh in self.flow_history]
        
        # Calculate trends
        magnitude_trend = 'increasing' if magnitudes[-1] > magnitudes[-2] else 'decreasing'
        if abs(magnitudes[-1] - magnitudes[-2]) < 0.5:
            magnitude_trend = 'stable'
        
        direction_change = abs(directions[-1] - directions[-2])
        if direction_change > math.pi:
            direction_change = 2 * math.pi - direction_change
        
        direction_trend = 'stable' if direction_change < 0.2 else 'changing'
        
        # Calculate acceleration (change in magnitude)
        if len(magnitudes) >= 3:
            flow_acceleration = magnitudes[-1] - 2*magnitudes[-2] + magnitudes[-3]
        else:
            flow_acceleration = 0
        
        return {
            'magnitude_trend': magnitude_trend,
            'direction_trend': direction_trend,
            'flow_acceleration': float(flow_acceleration),
            'direction_change': float(direction_change)
        }