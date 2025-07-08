"""
Behavior Analysis Model
Crowd behavior classification and anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque, defaultdict
import time
import logging
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

class CrowdBehavior(Enum):
    """Crowd behavior types"""
    NORMAL_FLOW = "normal_flow"
    CONGESTION = "congestion" 
    PANIC = "panic"
    GATHERING = "gathering"
    DISPERSING = "dispersing"
    QUEUING = "queuing"
    WANDERING = "wandering"
    BOTTLENECK = "bottleneck"
    COUNTER_FLOW = "counter_flow"
    ANOMALOUS = "anomalous"

class BehaviorLSTM(nn.Module):
    """LSTM network for temporal behavior analysis"""
    
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, num_classes=10):
        super(BehaviorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(attended_output)
        
        return output, attention_weights

class SpatialBehaviorCNN(nn.Module):
    """CNN for spatial behavior pattern recognition"""
    
    def __init__(self, num_classes=10):
        super(SpatialBehaviorCNN, self).__init__()
        
        # Spatial feature extractor
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class AnomalyDetector(nn.Module):
    """Autoencoder for crowd behavior anomaly detection"""
    
    def __init__(self, input_size=20, encoding_size=8):
        super(AnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_size),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class CrowdBehaviorAnalyzer:
    """Main class for crowd behavior analysis"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Models
        self.temporal_model = BehaviorLSTM().to(device)
        self.spatial_model = SpatialBehaviorCNN().to(device)
        self.anomaly_detector = AnomalyDetector().to(device)
        
        # Set to evaluation mode
        self.temporal_model.eval()
        self.spatial_model.eval()
        self.anomaly_detector.eval()
        
        # Feature history for temporal analysis
        self.feature_history = deque(maxlen=30)  # 30 frames
        self.behavior_history = deque(maxlen=100)
        
        # Clustering for behavior pattern discovery
        self.behavior_clusters = None
        self.scaler = StandardScaler()
        
        # Anomaly detection threshold
        self.anomaly_threshold = 0.1
        
    def extract_crowd_features(self, crowd_data):
        """Extract features from crowd analysis data"""
        density = crowd_data.get('density', {})
        movement = crowd_data.get('movement', {})
        tracks = crowd_data.get('tracks', [])
        
        features = {
            # Density features
            'overall_density': density.get('overall_density', 0),
            'max_local_density': density.get('max_local_density', 0),
            'avg_local_density': density.get('avg_local_density', 0),
            'density_variance': density.get('density_variance', 0),
            'hotspot_count': len(density.get('hotspots', [])),
            
            # Movement features
            'avg_speed': movement.get('avg_speed', 0),
            'max_speed': movement.get('max_speed', 0),
            'movement_ratio': movement.get('movement_ratio', 0),
            'flow_consistency': movement.get('flow_consistency', 1.0),
            'stationary_count': movement.get('stationary_count', 0),
            
            # Tracking features
            'total_tracks': len(tracks),
            'avg_track_length': np.mean([t.get('frames_tracked', 0) for t in tracks]) if tracks else 0,
            'track_density': len(tracks) / max(1, density.get('overall_density', 1)),
            
            # Spatial features
            'crowd_dispersion': self._calculate_dispersion(tracks),
            'interaction_strength': self._calculate_interactions(tracks),
            
            # Temporal features
            'density_trend': 1 if density.get('density_trend') == 'increasing' else -1 if density.get('density_trend') == 'decreasing' else 0,
            'trend_strength': density.get('trend_strength', 0),
            
            # Flow features
            'dominant_direction': movement.get('dominant_direction', 0),
            'flow_magnitude': movement.get('flow_magnitude', 0),
            'congestion_level': self._encode_congestion_level(movement.get('congestion_level', 'low')),
            
            # Additional metrics
            'crowd_coherence': self._calculate_coherence(tracks),
            'timestamp': time.time()
        }
        
        return features
    
    def _calculate_dispersion(self, tracks):
        """Calculate spatial dispersion of crowd"""
        if len(tracks) < 2:
            return 0
        
        positions = [track['centroid'] for track in tracks if 'centroid' in track]
        if len(positions) < 2:
            return 0
        
        # Calculate pairwise distances
        distances = pdist(positions)
        return float(np.mean(distances)) if len(distances) > 0 else 0
    
    def _calculate_interactions(self, tracks):
        """Calculate crowd interaction strength"""
        if len(tracks) < 2:
            return 0
        
        interaction_count = 0
        total_pairs = 0
        interaction_threshold = 50  # pixels
        
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks[i+1:], i+1):
                if 'centroid' in track1 and 'centroid' in track2:
                    distance = np.sqrt(
                        (track1['centroid'][0] - track2['centroid'][0])**2 +
                        (track1['centroid'][1] - track2['centroid'][1])**2
                    )
                    
                    if distance < interaction_threshold:
                        interaction_count += 1
                    total_pairs += 1
        
        return interaction_count / max(1, total_pairs)
    
    def _calculate_coherence(self, tracks):
        """Calculate movement coherence"""
        if len(tracks) < 2:
            return 0
        
        directions = []
        for track in tracks:
            movement = track.get('movement', {})
            if 'avg_direction' in movement:
                directions.append(movement['avg_direction'])
        
        if len(directions) < 2:
            return 0
        
        # Calculate circular variance of directions
        x_components = [np.cos(d) for d in directions]
        y_components = [np.sin(d) for d in directions]
        
        mean_x = np.mean(x_components)
        mean_y = np.mean(y_components)
        
        coherence = np.sqrt(mean_x**2 + mean_y**2)
        return float(coherence)
    
    def _encode_congestion_level(self, level):
        """Encode congestion level as numeric value"""
        levels = {'low': 0, 'medium': 1, 'high': 2}
        return levels.get(level, 0)
    
    def analyze_temporal_behavior(self, feature_sequence):
        """Analyze temporal behavior patterns"""
        if len(feature_sequence) < 5:  # Need minimum sequence length
            return {
                'behavior': CrowdBehavior.NORMAL_FLOW.value,
                'confidence': 0.5,
                'temporal_features': {}
            }
        
        # Convert to tensor
        feature_vector = []
        for features in feature_sequence:
            vector = [
                features['overall_density'],
                features['avg_speed'],
                features['flow_consistency'],
                features['movement_ratio'],
                features['density_trend'],
                features['trend_strength'],
                features['crowd_dispersion'],
                features['interaction_strength'],
                features['crowd_coherence'],
                features['congestion_level']
            ]
            feature_vector.append(vector)
        
        # Pad or truncate to fixed length
        sequence_length = 20
        if len(feature_vector) > sequence_length:
            feature_vector = feature_vector[-sequence_length:]
        else:
            # Pad with zeros
            padding = [[0] * len(feature_vector[0]) for _ in range(sequence_length - len(feature_vector))]
            feature_vector = padding + feature_vector
        
        input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output, attention = self.temporal_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to behavior enum
        behavior_types = list(CrowdBehavior)
        predicted_behavior = behavior_types[min(predicted_class, len(behavior_types)-1)]
        
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(feature_sequence)
        
        return {
            'behavior': predicted_behavior.value,
            'confidence': float(confidence),
            'temporal_features': temporal_features,
            'attention_weights': attention.squeeze().cpu().numpy().tolist()
        }
    
    def _calculate_temporal_features(self, feature_sequence):
        """Calculate additional temporal features"""
        if len(feature_sequence) < 2:
            return {}
        
        # Extract time series
        density_series = [f['overall_density'] for f in feature_sequence]
        speed_series = [f['avg_speed'] for f in feature_sequence]
        
        # Calculate trends and patterns
        density_changes = np.diff(density_series)
        speed_changes = np.diff(speed_series)
        
        return {
            'density_volatility': float(np.std(density_changes)) if len(density_changes) > 0 else 0,
            'speed_volatility': float(np.std(speed_changes)) if len(speed_changes) > 0 else 0,
            'avg_density_change': float(np.mean(density_changes)) if len(density_changes) > 0 else 0,
            'avg_speed_change': float(np.mean(speed_changes)) if len(speed_changes) > 0 else 0,
            'trend_consistency': self._calculate_trend_consistency(density_series),
            'periodicity': self._detect_periodicity(density_series)
        }
    
    def _calculate_trend_consistency(self, series):
        """Calculate how consistent the trend is"""
        if len(series) < 3:
            return 0
        
        changes = np.diff(series)
        if len(changes) == 0:
            return 0
        
        # Count sign changes
        sign_changes = sum(1 for i in range(len(changes)-1) 
                          if np.sign(changes[i]) != np.sign(changes[i+1]))
        
        consistency = 1.0 - (sign_changes / max(1, len(changes)-1))
        return float(consistency)
    
    def _detect_periodicity(self, series):
        """Detect periodic patterns in the series"""
        if len(series) < 6:
            return 0
        
        # Simple autocorrelation for periodicity detection
        try:
            autocorr = np.correlate(series, series, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks (excluding the first one which is always the highest)
            if len(autocorr) > 3:
                peaks = []
                for i in range(2, len(autocorr)-1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        peaks.append((i, autocorr[i]))
                
                if peaks:
                    # Return the strongest periodic component
                    strongest_peak = max(peaks, key=lambda x: x[1])
                    return float(strongest_peak[1] / autocorr[0])  # Normalized
            
            return 0
        except:
            return 0
    
    def detect_anomalies(self, features):
        """Detect anomalous behavior patterns"""
        # Prepare feature vector for anomaly detection
        feature_vector = [
            features['overall_density'],
            features['max_local_density'],
            features['avg_speed'],
            features['max_speed'],
            features['flow_consistency'],
            features['movement_ratio'],
            features['crowd_dispersion'],
            features['interaction_strength'],
            features['crowd_coherence'],
            features['density_variance'],
            features['hotspot_count'],
            features['total_tracks'],
            features['avg_track_length'],
            features['track_density'],
            features['trend_strength'],
            features['dominant_direction'],
            features['flow_magnitude'],
            features['congestion_level'],
            features['density_trend'],
            0  # Placeholder for additional feature
        ]
        
        input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstructed, encoded = self.anomaly_detector(input_tensor)
            reconstruction_error = F.mse_loss(reconstructed, input_tensor).item()
        
        is_anomaly = reconstruction_error > self.anomaly_threshold
        anomaly_score = float(reconstruction_error)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'threshold': self.anomaly_threshold,
            'encoded_features': encoded.squeeze().cpu().numpy().tolist()
        }
    
    def classify_crowd_state(self, features):
        """Classify current crowd state based on features"""
        density = features['overall_density']
        speed = features['avg_speed']
        consistency = features['flow_consistency']
        interaction = features['interaction_strength']
        coherence = features['crowd_coherence']
        congestion = features['congestion_level']
        
        # Rule-based classification
        if density > 25:
            if speed < 2:
                return CrowdBehavior.CONGESTION
            elif coherence < 0.3:
                return CrowdBehavior.PANIC
            else:
                return CrowdBehavior.GATHERING
        elif density < 5:
            if speed > 15:
                return CrowdBehavior.DISPERSING
            else:
                return CrowdBehavior.WANDERING
        else:
            if consistency < 0.3:
                if speed > 10:
                    return CrowdBehavior.COUNTER_FLOW
                else:
                    return CrowdBehavior.BOTTLENECK
            elif interaction > 0.7:
                return CrowdBehavior.QUEUING
            else:
                return CrowdBehavior.NORMAL_FLOW
    
    def analyze_behavior(self, crowd_data):
        """Main behavior analysis function"""
        # Extract features
        features = self.extract_crowd_features(crowd_data)
        
        # Add to history
        self.feature_history.append(features)
        
        # Temporal analysis
        temporal_result = self.analyze_temporal_behavior(list(self.feature_history))
        
        # Anomaly detection
        anomaly_result = self.detect_anomalies(features)
        
        # State classification
        crowd_state = self.classify_crowd_state(features)
        
        # Combine results
        analysis_result = {
            'timestamp': features['timestamp'],
            'crowd_state': crowd_state.value,
            'temporal_behavior': temporal_result,
            'anomaly_detection': anomaly_result,
            'features': features,
            'behavior_confidence': temporal_result['confidence'],
            'risk_assessment': self._assess_risk(features, temporal_result, anomaly_result)
        }
        
        # Store in behavior history
        self.behavior_history.append(analysis_result)
        
        return analysis_result
    
    def _assess_risk(self, features, temporal_result, anomaly_result):
        """Assess risk level based on behavior analysis"""
        risk_score = 0.0
        risk_factors = []
        
        # High density risk
        if features['overall_density'] > 20:
            risk_score += 0.3
            risk_factors.append("High crowd density")
        
        # Low flow consistency risk
        if features['flow_consistency'] < 0.3:
            risk_score += 0.2
            risk_factors.append("Poor flow consistency")
        
        # High speed risk (panic indicator)
        if features['avg_speed'] > 15:
            risk_score += 0.25
            risk_factors.append("High movement speed")
        
        # Anomaly risk
        if anomaly_result['is_anomaly']:
            risk_score += 0.3
            risk_factors.append("Anomalous behavior detected")
        
        # Dangerous behavior patterns
        dangerous_behaviors = [
            CrowdBehavior.PANIC.value,
            CrowdBehavior.BOTTLENECK.value,
            CrowdBehavior.COUNTER_FLOW.value
        ]
        
        if temporal_result['behavior'] in dangerous_behaviors:
            risk_score += 0.4
            risk_factors.append(f"Dangerous behavior: {temporal_result['behavior']}")
        
        # Rapid density changes
        if abs(features['trend_strength']) > 0.5:
            risk_score += 0.15
            risk_factors.append("Rapid density changes")
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = "CRITICAL"
        elif risk_score >= 0.6:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        elif risk_score >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'risk_level': risk_level,
            'risk_score': float(min(risk_score, 1.0)),
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(risk_level, risk_factors)
        }
    
    def _generate_recommendations(self, risk_level, risk_factors):
        """Generate safety recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED",
                "Activate emergency response procedures",
                "Consider immediate evacuation measures",
                "Deploy all available crowd control personnel"
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                "Increase monitoring and security presence",
                "Implement crowd flow management measures",
                "Prepare emergency response teams",
                "Consider limiting new entries"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Deploy additional staff to problem areas",
                "Implement preventive crowd management"
            ])
        elif risk_level == "LOW":
            recommendations.extend([
                "Continue normal monitoring",
                "Be alert for changes in crowd dynamics"
            ])
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if "High crowd density" in factor:
                recommendations.append("Reduce crowd density through controlled dispersal")
            elif "Poor flow consistency" in factor:
                recommendations.append("Improve crowd flow guidance and signage")
            elif "High movement speed" in factor:
                recommendations.append("Investigate cause of rapid movement - possible panic")
            elif "Anomalous behavior" in factor:
                recommendations.append("Investigate unusual crowd patterns")
            elif "Dangerous behavior" in factor:
                recommendations.append("Address dangerous crowd behavior immediately")
        
        return recommendations
    
    def get_behavior_trends(self, time_window=300):
        """Get behavior trends over specified time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_behaviors = [
            b for b in self.behavior_history
            if b['timestamp'] > cutoff_time
        ]
        
        if not recent_behaviors:
            return {}
        
        # Analyze trends
        behaviors = [b['crowd_state'] for b in recent_behaviors]
        risk_scores = [b['risk_assessment']['risk_score'] for b in recent_behaviors]
        
        behavior_counts = defaultdict(int)
        for behavior in behaviors:
            behavior_counts[behavior] += 1
        
        return {
            'time_window': time_window,
            'total_analyses': len(recent_behaviors),
            'behavior_distribution': dict(behavior_counts),
            'dominant_behavior': max(behavior_counts.items(), key=lambda x: x[1])[0],
            'avg_risk_score': float(np.mean(risk_scores)),
            'max_risk_score': float(np.max(risk_scores)),
            'risk_trend': 'increasing' if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else 'stable'
        }