"""
Alert System Module
Real-time alert generation and escalation based on crowd analysis
"""

import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
import asyncio
from typing import Dict, List, Optional, Callable

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class AlertType(Enum):
    """Types of alerts"""
    DENSITY_WARNING = "density_warning"
    MOVEMENT_ANOMALY = "movement_anomaly"
    CONGESTION = "congestion"
    RAPID_DENSITY_INCREASE = "rapid_density_increase"
    HOTSPOT_FORMATION = "hotspot_formation"
    FLOW_DISRUPTION = "flow_disruption"
    SAFETY_THRESHOLD = "safety_threshold"
    TECHNICAL_ISSUE = "technical_issue"

class Alert:
    """Individual alert object"""
    
    def __init__(self, alert_type: AlertType, level: AlertLevel, message: str, 
                 stream_id: str = None, location: str = None, data: dict = None):
        self.id = f"alert_{int(time.time() * 1000)}_{hash(message) % 10000}"
        self.alert_type = alert_type
        self.level = level
        self.message = message
        self.stream_id = stream_id
        self.location = location
        self.data = data or {}
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False
        self.escalated = False
        self.escalation_time = None
        self.resolution_time = None
        
    def to_dict(self):
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'type': self.alert_type.value,
            'level': self.level.name,
            'message': self.message,
            'stream_id': self.stream_id,
            'location': self.location,
            'data': self.data,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'escalated': self.escalated,
            'escalation_time': self.escalation_time,
            'resolution_time': self.resolution_time
        }
    
    def acknowledge(self):
        """Acknowledge the alert"""
        self.acknowledged = True
        
    def resolve(self):
        """Resolve the alert"""
        self.resolved = True
        self.resolution_time = time.time()
        
    def escalate(self):
        """Escalate the alert"""
        self.escalated = True
        self.escalation_time = time.time()
        if self.level.value < AlertLevel.EMERGENCY.value:
            self.level = AlertLevel(self.level.value + 1)

class AlertSystem:
    """Main alert system for crowd intelligence"""
    
    def __init__(self):
        self.active_alerts = {}  # Alert ID -> Alert
        self.alert_history = deque(maxlen=1000)  # Historical alerts
        self.subscribers = []  # Alert subscribers
        self.thresholds = self._default_thresholds()
        self.suppression_rules = {}  # Alert suppression rules
        self.escalation_rules = self._default_escalation_rules()
        
        # Alert frequency tracking
        self.alert_frequency = defaultdict(deque)
        self.rate_limits = {
            AlertType.DENSITY_WARNING: (5, 300),  # 5 alerts per 5 minutes
            AlertType.MOVEMENT_ANOMALY: (3, 180),  # 3 alerts per 3 minutes
            AlertType.CONGESTION: (3, 300),
            AlertType.HOTSPOT_FORMATION: (5, 600),
        }
        
        # Background processing
        self.running = False
        self.background_thread = None
        
    def _default_thresholds(self):
        """Default alert thresholds"""
        return {
            'density': {
                'low': 5.0,
                'medium': 10.0,
                'high': 15.0,
                'critical': 25.0,
                'emergency': 35.0
            },
            'movement_speed': {
                'slow': 2.0,
                'fast': 20.0
            },
            'congestion': {
                'medium': 0.3,  # Flow consistency threshold
                'high': 0.1
            },
            'hotspot_size': {
                'medium': 3,  # Number of people in hotspot
                'high': 8,
                'critical': 15
            }
        }
    
    def _default_escalation_rules(self):
        """Default escalation rules"""
        return {
            AlertLevel.LOW: 1800,      # 30 minutes
            AlertLevel.MEDIUM: 900,    # 15 minutes
            AlertLevel.HIGH: 300,      # 5 minutes
            AlertLevel.CRITICAL: 120,  # 2 minutes
        }
    
    def start(self):
        """Start the alert system background processing"""
        self.running = True
        self.background_thread = threading.Thread(target=self._background_processor)
        self.background_thread.daemon = True
        self.background_thread.start()
        logging.info("Alert system started")
    
    def stop(self):
        """Stop the alert system"""
        self.running = False
        if self.background_thread:
            self.background_thread.join()
        logging.info("Alert system stopped")
    
    def subscribe(self, callback: Callable[[Alert], None]):
        """Subscribe to alert notifications"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Alert], None]):
        """Unsubscribe from alert notifications"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def analyze_crowd_data(self, crowd_data: dict, stream_id: str = None, location: str = None):
        """Analyze crowd data and generate alerts"""
        alerts = []
        
        # Extract relevant metrics
        density = crowd_data.get('density', {})
        movement = crowd_data.get('movement', {})
        safety = crowd_data.get('safety_assessment', {})
        
        # Density-based alerts
        alerts.extend(self._check_density_alerts(density, stream_id, location))
        
        # Movement-based alerts
        alerts.extend(self._check_movement_alerts(movement, stream_id, location))
        
        # Safety-based alerts
        alerts.extend(self._check_safety_alerts(safety, stream_id, location))
        
        # Hotspot alerts
        alerts.extend(self._check_hotspot_alerts(density.get('hotspots', []), stream_id, location))
        
        # Process and fire alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def _check_density_alerts(self, density_data: dict, stream_id: str, location: str):
        """Check for density-related alerts"""
        alerts = []
        overall_density = density_data.get('overall_density', 0)
        trend = density_data.get('density_trend', 'stable')
        
        # Static density thresholds
        if overall_density >= self.thresholds['density']['emergency']:
            alerts.append(Alert(
                AlertType.DENSITY_WARNING,
                AlertLevel.EMERGENCY,
                f"EMERGENCY: Extremely high crowd density detected ({overall_density:.1f})",
                stream_id, location,
                {'density': overall_density, 'threshold': 'emergency'}
            ))
        elif overall_density >= self.thresholds['density']['critical']:
            alerts.append(Alert(
                AlertType.DENSITY_WARNING,
                AlertLevel.CRITICAL,
                f"CRITICAL: Very high crowd density detected ({overall_density:.1f})",
                stream_id, location,
                {'density': overall_density, 'threshold': 'critical'}
            ))
        elif overall_density >= self.thresholds['density']['high']:
            alerts.append(Alert(
                AlertType.DENSITY_WARNING,
                AlertLevel.HIGH,
                f"HIGH: High crowd density detected ({overall_density:.1f})",
                stream_id, location,
                {'density': overall_density, 'threshold': 'high'}
            ))
        elif overall_density >= self.thresholds['density']['medium']:
            alerts.append(Alert(
                AlertType.DENSITY_WARNING,
                AlertLevel.MEDIUM,
                f"Elevated crowd density detected ({overall_density:.1f})",
                stream_id, location,
                {'density': overall_density, 'threshold': 'medium'}
            ))
        
        # Rapid density increase
        if trend == 'increasing':
            trend_strength = density_data.get('trend_strength', 0)
            if trend_strength > 0.5:  # 50% increase
                alerts.append(Alert(
                    AlertType.RAPID_DENSITY_INCREASE,
                    AlertLevel.HIGH,
                    f"Rapid crowd density increase detected ({trend_strength:.1%} increase)",
                    stream_id, location,
                    {'trend_strength': trend_strength, 'current_density': overall_density}
                ))
        
        return alerts
    
    def _check_movement_alerts(self, movement_data: dict, stream_id: str, location: str):
        """Check for movement-related alerts"""
        alerts = []
        
        avg_speed = movement_data.get('avg_speed', 0)
        flow_consistency = movement_data.get('flow_consistency', 1.0)
        congestion_level = movement_data.get('congestion_level', 'low')
        
        # Movement speed anomalies
        if avg_speed < self.thresholds['movement_speed']['slow']:
            alerts.append(Alert(
                AlertType.MOVEMENT_ANOMALY,
                AlertLevel.MEDIUM,
                f"Abnormally slow crowd movement detected (avg: {avg_speed:.1f})",
                stream_id, location,
                {'avg_speed': avg_speed, 'type': 'slow_movement'}
            ))
        elif avg_speed > self.thresholds['movement_speed']['fast']:
            alerts.append(Alert(
                AlertType.MOVEMENT_ANOMALY,
                AlertLevel.HIGH,
                f"Abnormally fast crowd movement detected (avg: {avg_speed:.1f})",
                stream_id, location,
                {'avg_speed': avg_speed, 'type': 'fast_movement'}
            ))
        
        # Flow disruption
        if flow_consistency < self.thresholds['congestion']['high']:
            alerts.append(Alert(
                AlertType.FLOW_DISRUPTION,
                AlertLevel.HIGH,
                f"Severe crowd flow disruption detected (consistency: {flow_consistency:.2f})",
                stream_id, location,
                {'flow_consistency': flow_consistency, 'congestion_level': congestion_level}
            ))
        elif flow_consistency < self.thresholds['congestion']['medium']:
            alerts.append(Alert(
                AlertType.CONGESTION,
                AlertLevel.MEDIUM,
                f"Crowd congestion detected (consistency: {flow_consistency:.2f})",
                stream_id, location,
                {'flow_consistency': flow_consistency, 'congestion_level': congestion_level}
            ))
        
        return alerts
    
    def _check_safety_alerts(self, safety_data: dict, stream_id: str, location: str):
        """Check for safety-related alerts"""
        alerts = []
        
        safety_level = safety_data.get('safety_level', 'safe')
        risk_score = safety_data.get('risk_score', 0)
        recommendations = safety_data.get('recommendations', [])
        
        if safety_level == 'critical':
            alerts.append(Alert(
                AlertType.SAFETY_THRESHOLD,
                AlertLevel.CRITICAL,
                f"CRITICAL SAFETY ALERT: {', '.join(recommendations[:2])}",
                stream_id, location,
                {'safety_level': safety_level, 'risk_score': risk_score, 'recommendations': recommendations}
            ))
        elif safety_level == 'high_risk':
            alerts.append(Alert(
                AlertType.SAFETY_THRESHOLD,
                AlertLevel.HIGH,
                f"High risk safety condition detected (risk: {risk_score:.2f})",
                stream_id, location,
                {'safety_level': safety_level, 'risk_score': risk_score}
            ))
        elif safety_level == 'caution':
            alerts.append(Alert(
                AlertType.SAFETY_THRESHOLD,
                AlertLevel.MEDIUM,
                f"Caution: Elevated safety risk (risk: {risk_score:.2f})",
                stream_id, location,
                {'safety_level': safety_level, 'risk_score': risk_score}
            ))
        
        return alerts
    
    def _check_hotspot_alerts(self, hotspots: list, stream_id: str, location: str):
        """Check for hotspot-related alerts"""
        alerts = []
        
        for hotspot in hotspots:
            density = hotspot.get('density', 0)
            severity = hotspot.get('severity', 'low')
            coordinates = hotspot.get('coordinates', [])
            
            if density >= self.thresholds['hotspot_size']['critical']:
                alerts.append(Alert(
                    AlertType.HOTSPOT_FORMATION,
                    AlertLevel.CRITICAL,
                    f"Critical density hotspot formed ({int(density)} people)",
                    stream_id, location,
                    {'hotspot_density': density, 'coordinates': coordinates, 'severity': severity}
                ))
            elif density >= self.thresholds['hotspot_size']['high']:
                alerts.append(Alert(
                    AlertType.HOTSPOT_FORMATION,
                    AlertLevel.HIGH,
                    f"High density hotspot detected ({int(density)} people)",
                    stream_id, location,
                    {'hotspot_density': density, 'coordinates': coordinates, 'severity': severity}
                ))
            elif density >= self.thresholds['hotspot_size']['medium']:
                alerts.append(Alert(
                    AlertType.HOTSPOT_FORMATION,
                    AlertLevel.MEDIUM,
                    f"Density hotspot forming ({int(density)} people)",
                    stream_id, location,
                    {'hotspot_density': density, 'coordinates': coordinates, 'severity': severity}
                ))
        
        return alerts
    
    def _process_alert(self, alert: Alert):
        """Process and potentially fire an alert"""
        # Check rate limiting
        if not self._check_rate_limit(alert):
            return
        
        # Check suppression rules
        if self._is_suppressed(alert):
            return
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Update frequency tracking
        self.alert_frequency[alert.alert_type].append(time.time())
        
        # Notify subscribers
        self._notify_subscribers(alert)
        
        logging.info(f"Alert fired: {alert.level.name} - {alert.message}")
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert exceeds rate limits"""
        if alert.alert_type not in self.rate_limits:
            return True
        
        max_alerts, time_window = self.rate_limits[alert.alert_type]
        current_time = time.time()
        
        # Clean old entries
        frequency_queue = self.alert_frequency[alert.alert_type]
        while frequency_queue and current_time - frequency_queue[0] > time_window:
            frequency_queue.popleft()
        
        # Check if under limit
        return len(frequency_queue) < max_alerts
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        # Simple suppression: don't repeat identical alerts within time window
        suppression_window = 300  # 5 minutes
        current_time = time.time()
        
        for existing_alert in self.active_alerts.values():
            if (existing_alert.alert_type == alert.alert_type and
                existing_alert.stream_id == alert.stream_id and
                existing_alert.level == alert.level and
                not existing_alert.resolved and
                current_time - existing_alert.timestamp < suppression_window):
                return True
        
        return False
    
    def _notify_subscribers(self, alert: Alert):
        """Notify all subscribers of new alert"""
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error notifying alert subscriber: {e}")
    
    def _background_processor(self):
        """Background thread for alert processing"""
        while self.running:
            try:
                self._process_escalations()
                self._cleanup_resolved_alerts()
                time.sleep(30)  # Run every 30 seconds
            except Exception as e:
                logging.error(f"Error in alert background processor: {e}")
    
    def _process_escalations(self):
        """Process alert escalations"""
        current_time = time.time()
        
        for alert in list(self.active_alerts.values()):
            if (not alert.acknowledged and 
                not alert.escalated and 
                not alert.resolved and
                alert.level in self.escalation_rules):
                
                escalation_time = self.escalation_rules[alert.level]
                if current_time - alert.timestamp > escalation_time:
                    alert.escalate()
                    self._notify_subscribers(alert)
                    logging.warning(f"Alert escalated: {alert.id} to {alert.level.name}")
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if (alert.resolved and 
                current_time - alert.resolution_time > cleanup_age):
                to_remove.append(alert_id)
        
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
    
    def acknowledge_alert(self, alert_id: str, user: str = None):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge()
            if user:
                alert.data['acknowledged_by'] = user
            logging.info(f"Alert acknowledged: {alert_id} by {user}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = None, resolution_note: str = None):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()
            if user:
                alert.data['resolved_by'] = user
            if resolution_note:
                alert.data['resolution_note'] = resolution_note
            logging.info(f"Alert resolved: {alert_id} by {user}")
            return True
        return False
    
    def get_active_alerts(self, level: AlertLevel = None, alert_type: AlertType = None):
        """Get active alerts with optional filtering"""
        alerts = []
        for alert in self.active_alerts.values():
            if not alert.resolved:
                if level and alert.level != level:
                    continue
                if alert_type and alert.alert_type != alert_type:
                    continue
                alerts.append(alert.to_dict())
        
        # Sort by level (highest first) then by timestamp (newest first)
        alerts.sort(key=lambda x: (-(AlertLevel[x['level']].value), -x['timestamp']))
        return alerts
    
    def get_alert_statistics(self, time_window: int = 3600):
        """Get alert statistics for the specified time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
        
        stats = {
            'total_alerts': len(recent_alerts),
            'by_level': defaultdict(int),
            'by_type': defaultdict(int),
            'by_stream': defaultdict(int),
            'resolved_count': 0,
            'acknowledged_count': 0,
            'escalated_count': 0,
            'avg_resolution_time': 0
        }
        
        resolution_times = []
        
        for alert in recent_alerts:
            stats['by_level'][alert.level.name] += 1
            stats['by_type'][alert.alert_type.value] += 1
            if alert.stream_id:
                stats['by_stream'][alert.stream_id] += 1
            
            if alert.resolved:
                stats['resolved_count'] += 1
                if alert.resolution_time:
                    resolution_times.append(alert.resolution_time - alert.timestamp)
            
            if alert.acknowledged:
                stats['acknowledged_count'] += 1
            
            if alert.escalated:
                stats['escalated_count'] += 1
        
        if resolution_times:
            stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)
        
        return stats
    
    def create_manual_alert(self, alert_type: AlertType, level: AlertLevel, 
                          message: str, stream_id: str = None, location: str = None,
                          data: dict = None, user: str = None):
        """Create a manual alert"""
        alert = Alert(alert_type, level, message, stream_id, location, data)
        if user:
            alert.data['created_by'] = user
        
        self._process_alert(alert)
        return alert.id