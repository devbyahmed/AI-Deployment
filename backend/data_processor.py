"""
Data Processor
Handles data transformation, aggregation, and export for crowd intelligence
"""

import pandas as pd
import numpy as np
import json
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and transform crowd intelligence data"""
    
    def __init__(self):
        pass
    
    def process_crowd_data(self, raw_data: Dict) -> Dict:
        """Process raw crowd detection data"""
        processed = {
            "timestamp": raw_data.get("timestamp", datetime.now().timestamp()),
            "person_count": raw_data.get("count", 0),
            "confidence_avg": raw_data.get("confidence", 0),
            "density_score": raw_data.get("density", 0),
            "processing_time_ms": raw_data.get("processing_time", 0) * 1000
        }
        
        # Calculate derived metrics
        if "persons" in raw_data:
            persons = raw_data["persons"]
            if persons:
                confidences = [p.get("confidence", 0) for p in persons]
                processed.update({
                    "confidence_min": min(confidences),
                    "confidence_max": max(confidences),
                    "confidence_std": np.std(confidences)
                })
        
        return processed
    
    def aggregate_analytics(self, analytics_list: List[Dict], 
                          time_window: str = "hour") -> Dict:
        """Aggregate analytics data over time windows"""
        if not analytics_list:
            return {}
        
        df = pd.DataFrame(analytics_list)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Define aggregation rules
        agg_rules = {
            'person_count': ['mean', 'max', 'min', 'std'],
            'density_score': ['mean', 'max', 'min'],
            'processing_time_ms': ['mean', 'max'],
            'confidence_avg': ['mean']
        }
        
        # Resample based on time window
        if time_window == "minute":
            resampled = df.resample('1T').agg(agg_rules)
        elif time_window == "hour":
            resampled = df.resample('1H').agg(agg_rules)
        elif time_window == "day":
            resampled = df.resample('1D').agg(agg_rules)
        else:
            resampled = df.agg(agg_rules)
        
        # Convert to dictionary
        result = {}
        for col, stats in agg_rules.items():
            result[col] = {}
            for stat in stats:
                if isinstance(resampled, pd.DataFrame):
                    values = resampled[(col, stat)].dropna().to_dict()
                    result[col][stat] = {
                        str(k): float(v) for k, v in values.items()
                    }
                else:
                    result[col][stat] = float(resampled[(col, stat)])
        
        return result
    
    def calculate_crowd_trends(self, analytics_list: List[Dict]) -> Dict:
        """Calculate crowd trends and patterns"""
        if len(analytics_list) < 2:
            return {"trend": "insufficient_data"}
        
        df = pd.DataFrame(analytics_list)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.sort_values('datetime', inplace=True)
        
        trends = {}
        
        # Calculate trends for key metrics
        for metric in ['person_count', 'density_score']:
            if metric in df.columns:
                values = df[metric].values
                
                # Linear trend
                x = np.arange(len(values))
                if len(values) > 1:
                    slope = np.polyfit(x, values, 1)[0]
                    
                    trends[metric] = {
                        "slope": float(slope),
                        "direction": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable",
                        "current_value": float(values[-1]),
                        "change_rate": float((values[-1] - values[0]) / len(values)) if len(values) > 1 else 0,
                        "volatility": float(np.std(values))
                    }
        
        # Peak detection
        if 'person_count' in df.columns:
            counts = df['person_count'].values
            peaks = self._find_peaks(counts)
            trends['peaks'] = {
                "count": len(peaks),
                "peak_times": [str(df.iloc[i]['datetime']) for i in peaks],
                "peak_values": [float(counts[i]) for i in peaks]
            }
        
        return trends
    
    def _find_peaks(self, data: np.ndarray, prominence: float = 0.1) -> List[int]:
        """Simple peak detection algorithm"""
        peaks = []
        if len(data) < 3:
            return peaks
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # Check prominence
                left_min = min(data[max(0, i-5):i])
                right_min = min(data[i+1:min(len(data), i+6)])
                if data[i] - max(left_min, right_min) > prominence * max(data):
                    peaks.append(i)
        
        return peaks
    
    def generate_report(self, stream_data: Dict, time_range: Dict) -> Dict:
        """Generate comprehensive analytics report"""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_range": time_range,
                "stream_id": stream_data.get("stream_id"),
                "location": stream_data.get("location")
            },
            "summary": {},
            "detailed_analytics": {},
            "alerts_summary": {},
            "recommendations": []
        }
        
        analytics = stream_data.get("analytics", [])
        if analytics:
            # Summary statistics
            df = pd.DataFrame(analytics)
            
            report["summary"] = {
                "total_records": len(analytics),
                "avg_crowd_count": float(df['person_count'].mean()) if 'person_count' in df else 0,
                "max_crowd_count": float(df['person_count'].max()) if 'person_count' in df else 0,
                "avg_density": float(df['density_score'].mean()) if 'density_score' in df else 0,
                "peak_periods": self._identify_peak_periods(analytics)
            }
            
            # Trends
            report["detailed_analytics"]["trends"] = self.calculate_crowd_trends(analytics)
            
            # Hourly patterns
            report["detailed_analytics"]["hourly_patterns"] = self._analyze_hourly_patterns(analytics)
        
        # Alerts summary
        alerts = stream_data.get("alerts", [])
        if alerts:
            alert_df = pd.DataFrame(alerts)
            report["alerts_summary"] = {
                "total_alerts": len(alerts),
                "critical_alerts": len(alert_df[alert_df['level'] == 'CRITICAL']) if 'level' in alert_df else 0,
                "alert_types": alert_df['type'].value_counts().to_dict() if 'type' in alert_df else {},
                "resolution_rate": (alert_df['resolved'].sum() / len(alerts)) if 'resolved' in alert_df else 0
            }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["summary"], report["alerts_summary"])
        
        return report
    
    def _identify_peak_periods(self, analytics: List[Dict]) -> List[Dict]:
        """Identify peak crowd periods"""
        if not analytics:
            return []
        
        df = pd.DataFrame(analytics)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Find periods with high crowd density
        threshold = df['person_count'].quantile(0.8) if 'person_count' in df else 0
        peak_periods = []
        
        high_density_periods = df[df['person_count'] > threshold] if 'person_count' in df else pd.DataFrame()
        
        if not high_density_periods.empty:
            # Group consecutive high-density periods
            high_density_periods = high_density_periods.sort_values('datetime')
            
            current_period_start = None
            current_period_end = None
            
            for _, row in high_density_periods.iterrows():
                if current_period_start is None:
                    current_period_start = row['datetime']
                    current_period_end = row['datetime']
                elif (row['datetime'] - current_period_end).seconds <= 1800:  # Within 30 minutes
                    current_period_end = row['datetime']
                else:
                    # End current period, start new one
                    peak_periods.append({
                        "start": current_period_start.isoformat(),
                        "end": current_period_end.isoformat(),
                        "duration_minutes": (current_period_end - current_period_start).seconds / 60,
                        "max_count": float(df[(df['datetime'] >= current_period_start) & 
                                            (df['datetime'] <= current_period_end)]['person_count'].max())
                    })
                    current_period_start = row['datetime']
                    current_period_end = row['datetime']
            
            # Don't forget the last period
            if current_period_start is not None:
                peak_periods.append({
                    "start": current_period_start.isoformat(),
                    "end": current_period_end.isoformat(),
                    "duration_minutes": (current_period_end - current_period_start).seconds / 60,
                    "max_count": float(df[(df['datetime'] >= current_period_start) & 
                                        (df['datetime'] <= current_period_end)]['person_count'].max())
                })
        
        return peak_periods
    
    def _analyze_hourly_patterns(self, analytics: List[Dict]) -> Dict:
        """Analyze crowd patterns by hour of day"""
        if not analytics:
            return {}
        
        df = pd.DataFrame(analytics)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        
        hourly_stats = df.groupby('hour')['person_count'].agg(['mean', 'max', 'count']).to_dict()
        
        return {
            "avg_by_hour": {str(k): float(v) for k, v in hourly_stats['mean'].items()},
            "max_by_hour": {str(k): float(v) for k, v in hourly_stats['max'].items()},
            "data_points_by_hour": {str(k): int(v) for k, v in hourly_stats['count'].items()}
        }
    
    def _generate_recommendations(self, summary: Dict, alerts_summary: Dict) -> List[str]:
        """Generate actionable recommendations based on data"""
        recommendations = []
        
        avg_crowd = summary.get("avg_crowd_count", 0)
        max_crowd = summary.get("max_crowd_count", 0)
        critical_alerts = alerts_summary.get("critical_alerts", 0)
        total_alerts = alerts_summary.get("total_alerts", 0)
        
        # Crowd management recommendations
        if max_crowd > 50:
            recommendations.append("Consider implementing crowd flow management during peak periods")
        
        if avg_crowd > 30:
            recommendations.append("Monitor crowd density closely - approaching capacity limits")
        
        # Alert-based recommendations
        if critical_alerts > 0:
            recommendations.append(f"Address {critical_alerts} critical safety alert(s) immediately")
        
        if total_alerts > 10:
            recommendations.append("High alert frequency detected - review safety protocols")
        
        # Performance recommendations
        if summary.get("peak_periods") and len(summary["peak_periods"]) > 3:
            recommendations.append("Implement predictive crowd management for recurring peak periods")
        
        return recommendations
    
    def to_csv(self, data: List[Dict]) -> str:
        """Convert data to CSV format"""
        if not data:
            return ""
        
        output = io.StringIO()
        
        # Flatten nested dictionaries
        flattened_data = []
        for item in data:
            flattened = self._flatten_dict(item)
            flattened_data.append(flattened)
        
        if flattened_data:
            writer = csv.DictWriter(output, fieldnames=flattened_data[0].keys())
            writer.writeheader()
            writer.writerows(flattened_data)
        
        return output.getvalue()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def calculate_kpis(self, analytics_data: List[Dict]) -> Dict:
        """Calculate key performance indicators"""
        if not analytics_data:
            return {}
        
        df = pd.DataFrame(analytics_data)
        
        kpis = {
            "utilization": {
                "avg_occupancy": float(df['person_count'].mean()) if 'person_count' in df else 0,
                "peak_occupancy": float(df['person_count'].max()) if 'person_count' in df else 0,
                "utilization_rate": 0  # Would need capacity data
            },
            "safety": {
                "avg_density": float(df['density_score'].mean()) if 'density_score' in df else 0,
                "safety_incidents": 0,  # Would come from alerts
                "compliance_rate": 0.95  # Placeholder
            },
            "performance": {
                "system_uptime": 0.99,  # Placeholder
                "avg_processing_time": float(df['processing_time_ms'].mean()) if 'processing_time_ms' in df else 0,
                "accuracy": float(df['confidence_avg'].mean()) if 'confidence_avg' in df else 0
            }
        }
        
        return kpis
    
    def export_dashboard_data(self, stream_data: Dict) -> Dict:
        """Export data formatted for dashboard consumption"""
        dashboard_data = {
            "real_time": {
                "current_count": 0,
                "current_density": 0,
                "alert_level": "normal",
                "last_updated": datetime.now().isoformat()
            },
            "charts": {
                "hourly_trends": [],
                "density_heatmap": [],
                "alert_timeline": []
            },
            "statistics": {
                "today_peak": 0,
                "avg_occupancy": 0,
                "total_alerts": 0
            }
        }
        
        # Process recent data for real-time display
        recent_analytics = stream_data.get("recent_analytics", [])
        if recent_analytics:
            latest = recent_analytics[0]
            dashboard_data["real_time"]["current_count"] = latest.get("person_count", 0)
            dashboard_data["real_time"]["current_density"] = latest.get("density_score", 0)
        
        # Process historical data for charts
        all_analytics = stream_data.get("analytics", [])
        if all_analytics:
            # Hourly trends
            hourly_patterns = self._analyze_hourly_patterns(all_analytics)
            dashboard_data["charts"]["hourly_trends"] = [
                {"hour": int(h), "count": v} 
                for h, v in hourly_patterns.get("avg_by_hour", {}).items()
            ]
            
            # Statistics
            df = pd.DataFrame(all_analytics)
            dashboard_data["statistics"]["today_peak"] = float(df['person_count'].max()) if 'person_count' in df else 0
            dashboard_data["statistics"]["avg_occupancy"] = float(df['person_count'].mean()) if 'person_count' in df else 0
        
        return dashboard_data