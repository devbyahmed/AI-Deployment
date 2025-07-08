"""
Density Calculator Module
Advanced crowd density algorithms and spatial analysis
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import distance
import math
from collections import defaultdict
import logging
import time

class DensityCalculator:
    """Advanced crowd density calculation using multiple algorithms"""
    
    def __init__(self, frame_shape=None):
        self.frame_shape = frame_shape
        self.grid_size = (20, 20)  # Default grid for spatial analysis
        self.density_history = []
        self.max_history = 100
        
        # Perspective correction parameters
        self.perspective_matrix = None
        self.calibration_points = None
        
        # Density calculation methods
        self.methods = {
            'grid_based': self.grid_based_density,
            'kernel_density': self.kernel_density_estimation,
            'voronoi': self.voronoi_density,
            'perspective_corrected': self.perspective_corrected_density
        }
    
    def set_perspective_calibration(self, real_world_points, image_points):
        """Set perspective correction for accurate density calculation"""
        try:
            real_world_points = np.array(real_world_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            
            self.perspective_matrix = cv2.getPerspectiveTransform(
                image_points, real_world_points
            )
            self.calibration_points = {
                'real_world': real_world_points,
                'image': image_points
            }
            
            logging.info("Perspective calibration set successfully")
        except Exception as e:
            logging.error(f"Error setting perspective calibration: {e}")
            self.perspective_matrix = None
    
    def calculate_density(self, detections, frame_shape=None, method='grid_based'):
        """Calculate crowd density using specified method"""
        if frame_shape:
            self.frame_shape = frame_shape
        
        if not detections or not self.frame_shape:
            return self._empty_density_result()
        
        # Extract person positions
        positions = []
        confidences = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            positions.append(center)
            confidences.append(detection.get('confidence', 1.0))
        
        # Calculate density using specified method
        if method in self.methods:
            density_result = self.methods[method](positions, confidences)
        else:
            density_result = self.grid_based_density(positions, confidences)
        
        # Add temporal analysis
        density_result.update(self._calculate_temporal_trends())
        
        # Store in history
        self.density_history.append({
            'timestamp': time.time(),
            'density': density_result['overall_density'],
            'count': len(positions)
        })
        
        # Maintain history size
        if len(self.density_history) > self.max_history:
            self.density_history.pop(0)
        
        return density_result
    
    def grid_based_density(self, positions, confidences):
        """Grid-based density calculation"""
        height, width = self.frame_shape[:2]
        grid_h, grid_w = self.grid_size
        
        # Initialize density grid
        density_grid = np.zeros(self.grid_size)
        confidence_grid = np.zeros(self.grid_size)
        
        # Map positions to grid cells
        for pos, conf in zip(positions, confidences):
            x, y = pos
            
            # Convert to grid coordinates
            grid_x = min(int(x / width * grid_w), grid_w - 1)
            grid_y = min(int(y / height * grid_h), grid_h - 1)
            
            density_grid[grid_y, grid_x] += 1
            confidence_grid[grid_y, grid_x] += conf
        
        # Calculate metrics
        total_area = width * height
        cell_area = total_area / (grid_h * grid_w)
        
        # Overall density (people per unit area)
        overall_density = len(positions) / total_area * 10000  # per 10k pixels
        
        # Find hotspots
        hotspots = self._find_hotspots(density_grid, grid_h, grid_w, width, height)
        
        # Calculate local densities
        local_densities = density_grid / cell_area * 10000
        
        return {
            'method': 'grid_based',
            'overall_density': float(overall_density),
            'density_grid': density_grid.tolist(),
            'local_densities': local_densities.tolist(),
            'confidence_grid': confidence_grid.tolist(),
            'hotspots': hotspots,
            'max_local_density': float(np.max(local_densities)),
            'avg_local_density': float(np.mean(local_densities)),
            'density_variance': float(np.var(local_densities)),
            'occupied_cells': int(np.count_nonzero(density_grid)),
            'total_cells': grid_h * grid_w
        }
    
    def kernel_density_estimation(self, positions, confidences):
        """Kernel density estimation for smooth density distribution"""
        if not positions:
            return self._empty_density_result()
        
        height, width = self.frame_shape[:2]
        
        # Create coordinate grids
        x = np.linspace(0, width, 100)
        y = np.linspace(0, height, 100)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Kernel density estimation using Gaussian kernels
        bandwidth = min(width, height) * 0.05  # Adaptive bandwidth
        density_map = np.zeros(grid_points.shape[0])
        
        for pos, conf in zip(positions, confidences):
            # Calculate Gaussian kernel for each person
            distances = np.sqrt(np.sum((grid_points - np.array(pos))**2, axis=1))
            kernel_values = conf * np.exp(-0.5 * (distances / bandwidth)**2)
            density_map += kernel_values
        
        # Reshape to 2D
        density_map = density_map.reshape(X.shape)
        
        # Normalize
        density_map = density_map / (2 * np.pi * bandwidth**2)
        
        # Calculate metrics
        overall_density = len(positions) / (width * height) * 10000
        max_density = float(np.max(density_map))
        
        # Find peaks (hotspots)
        peaks = self._find_density_peaks(density_map, x, y)
        
        return {
            'method': 'kernel_density',
            'overall_density': float(overall_density),
            'density_map': density_map.tolist(),
            'max_density': max_density,
            'avg_density': float(np.mean(density_map)),
            'density_variance': float(np.var(density_map)),
            'hotspots': peaks,
            'bandwidth': bandwidth
        }
    
    def voronoi_density(self, positions, confidences):
        """Voronoi diagram-based density calculation"""
        if len(positions) < 3:
            return self._empty_density_result()
        
        try:
            from scipy.spatial import Voronoi
            
            height, width = self.frame_shape[:2]
            positions_array = np.array(positions)
            
            # Add boundary points to ensure finite regions
            boundary_points = [
                [0, 0], [width, 0], [width, height], [0, height],
                [width/2, 0], [width, height/2], [width/2, height], [0, height/2]
            ]
            
            all_points = np.vstack([positions_array, boundary_points])
            
            # Compute Voronoi diagram
            vor = Voronoi(all_points)
            
            # Calculate density for each region
            densities = []
            areas = []
            
            for i, point in enumerate(positions_array):
                region_index = vor.point_region[i]
                region = vor.regions[region_index]
                
                if -1 not in region and len(region) > 0:
                    # Calculate region area
                    vertices = vor.vertices[region]
                    area = self._polygon_area(vertices)
                    
                    if area > 0:
                        density = 1.0 / area * 10000  # Inverse area as density
                        densities.append(density)
                        areas.append(area)
                    else:
                        densities.append(0)
                        areas.append(0)
                else:
                    densities.append(0)
                    areas.append(0)
            
            overall_density = len(positions) / (width * height) * 10000
            
            return {
                'method': 'voronoi',
                'overall_density': float(overall_density),
                'local_densities': densities,
                'region_areas': areas,
                'max_local_density': float(max(densities)) if densities else 0,
                'avg_local_density': float(np.mean(densities)) if densities else 0,
                'density_variance': float(np.var(densities)) if densities else 0,
                'num_regions': len(densities)
            }
            
        except ImportError:
            logging.warning("Scipy not available for Voronoi density calculation")
            return self.grid_based_density(positions, confidences)
        except Exception as e:
            logging.error(f"Error in Voronoi density calculation: {e}")
            return self.grid_based_density(positions, confidences)
    
    def perspective_corrected_density(self, positions, confidences):
        """Perspective-corrected density calculation"""
        if self.perspective_matrix is None:
            logging.warning("No perspective calibration available, using grid-based method")
            return self.grid_based_density(positions, confidences)
        
        # Transform positions to real-world coordinates
        real_world_positions = []
        for pos in positions:
            # Convert to homogeneous coordinates
            point = np.array([[pos[0], pos[1]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(
                point.reshape(-1, 1, 2), self.perspective_matrix
            )
            real_world_positions.append(tuple(transformed[0, 0]))
        
        # Calculate density in real-world coordinates
        if not real_world_positions:
            return self._empty_density_result()
        
        # Find bounds of real-world area
        x_coords = [pos[0] for pos in real_world_positions]
        y_coords = [pos[1] for pos in real_world_positions]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        real_world_area = (max_x - min_x) * (max_y - min_y)
        
        # Density calculation (people per square meter if calibrated in meters)
        overall_density = len(positions) / real_world_area if real_world_area > 0 else 0
        
        # Grid-based analysis in real-world coordinates
        grid_size = 20
        x_bins = np.linspace(min_x, max_x, grid_size)
        y_bins = np.linspace(min_y, max_y, grid_size)
        
        density_grid = np.zeros((grid_size-1, grid_size-1))
        
        for pos in real_world_positions:
            x_idx = np.digitize(pos[0], x_bins) - 1
            y_idx = np.digitize(pos[1], y_bins) - 1
            
            if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
                density_grid[y_idx, x_idx] += 1
        
        # Calculate cell areas in real-world units
        cell_width = (max_x - min_x) / (grid_size - 1)
        cell_height = (max_y - min_y) / (grid_size - 1)
        cell_area = cell_width * cell_height
        
        local_densities = density_grid / cell_area if cell_area > 0 else density_grid
        
        return {
            'method': 'perspective_corrected',
            'overall_density': float(overall_density),
            'density_grid': density_grid.tolist(),
            'local_densities': local_densities.tolist(),
            'real_world_area': float(real_world_area),
            'cell_area': float(cell_area),
            'max_local_density': float(np.max(local_densities)),
            'avg_local_density': float(np.mean(local_densities)),
            'density_variance': float(np.var(local_densities)),
            'calibration_used': True
        }
    
    def _find_hotspots(self, density_grid, grid_h, grid_w, width, height):
        """Find density hotspots in the grid"""
        hotspots = []
        
        # Use statistical threshold for hotspot detection
        mean_density = np.mean(density_grid)
        std_density = np.std(density_grid)
        threshold = mean_density + 2 * std_density
        
        for i in range(grid_h):
            for j in range(grid_w):
                if density_grid[i, j] > threshold and density_grid[i, j] > 0:
                    # Convert grid coordinates back to image coordinates
                    x1 = int(j * width / grid_w)
                    y1 = int(i * height / grid_h)
                    x2 = int((j + 1) * width / grid_w)
                    y2 = int((i + 1) * height / grid_h)
                    
                    hotspots.append({
                        'grid_position': (i, j),
                        'density': float(density_grid[i, j]),
                        'coordinates': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'severity': 'high' if density_grid[i, j] > threshold * 1.5 else 'medium'
                    })
        
        # Sort by density
        hotspots.sort(key=lambda x: x['density'], reverse=True)
        
        return hotspots
    
    def _find_density_peaks(self, density_map, x, y):
        """Find peaks in kernel density map"""
        from scipy.ndimage import maximum_filter
        
        # Find local maxima
        neighborhood = np.ones((3, 3))
        local_maxima = maximum_filter(density_map, footprint=neighborhood) == density_map
        
        # Threshold for significant peaks
        threshold = np.mean(density_map) + 2 * np.std(density_map)
        significant_peaks = (density_map > threshold) & local_maxima
        
        peaks = []
        peak_coords = np.where(significant_peaks)
        
        for i, j in zip(peak_coords[0], peak_coords[1]):
            peaks.append({
                'position': (float(x[j]), float(y[i])),
                'density': float(density_map[i, j]),
                'grid_position': (i, j)
            })
        
        return peaks
    
    def _polygon_area(self, vertices):
        """Calculate area of polygon using shoelace formula"""
        if len(vertices) < 3:
            return 0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                            for i in range(-1, len(x)-1)))
        return area
    
    def _calculate_temporal_trends(self):
        """Calculate temporal trends in density"""
        if len(self.density_history) < 2:
            return {
                'density_trend': 'stable',
                'trend_strength': 0,
                'peak_density': 0,
                'avg_density': 0
            }
        
        densities = [h['density'] for h in self.density_history[-10:]]  # Last 10 measurements
        
        # Calculate trend
        if len(densities) >= 3:
            recent_avg = np.mean(densities[-3:])
            earlier_avg = np.mean(densities[:3])
            
            trend_strength = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            
            if abs(trend_strength) < 0.1:
                trend = 'stable'
            elif trend_strength > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'stable'
            trend_strength = 0
        
        return {
            'density_trend': trend,
            'trend_strength': float(trend_strength),
            'peak_density': float(max([h['density'] for h in self.density_history])),
            'avg_density': float(np.mean([h['density'] for h in self.density_history])),
            'density_volatility': float(np.std([h['density'] for h in self.density_history[-10:]]))
        }
    
    def _empty_density_result(self):
        """Return empty density result"""
        return {
            'method': 'none',
            'overall_density': 0,
            'density_grid': [],
            'local_densities': [],
            'hotspots': [],
            'max_local_density': 0,
            'avg_local_density': 0,
            'density_variance': 0,
            'density_trend': 'stable',
            'trend_strength': 0
        }
    
    def get_safety_assessment(self, density_result):
        """Assess safety level based on density metrics"""
        overall_density = density_result.get('overall_density', 0)
        hotspots = density_result.get('hotspots', [])
        trend = density_result.get('density_trend', 'stable')
        
        # Safety thresholds (people per 10k pixels)
        # These would be calibrated based on venue type and safety standards
        safe_threshold = 5.0
        caution_threshold = 10.0
        danger_threshold = 20.0
        
        # Base safety level
        if overall_density < safe_threshold:
            safety_level = 'safe'
            risk_score = 0.2
        elif overall_density < caution_threshold:
            safety_level = 'caution'
            risk_score = 0.5
        elif overall_density < danger_threshold:
            safety_level = 'high_risk'
            risk_score = 0.8
        else:
            safety_level = 'critical'
            risk_score = 1.0
        
        # Adjust for hotspots
        high_severity_hotspots = len([h for h in hotspots if h.get('severity') == 'high'])
        if high_severity_hotspots > 0:
            risk_score = min(1.0, risk_score + 0.2 * high_severity_hotspots)
        
        # Adjust for trend
        if trend == 'increasing':
            risk_score = min(1.0, risk_score + 0.1)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(
            safety_level, overall_density, hotspots, trend
        )
        
        return {
            'safety_level': safety_level,
            'risk_score': float(risk_score),
            'overall_density': float(overall_density),
            'hotspot_count': len(hotspots),
            'high_risk_areas': high_severity_hotspots,
            'trend': trend,
            'recommendations': recommendations
        }
    
    def _generate_safety_recommendations(self, safety_level, density, hotspots, trend):
        """Generate safety recommendations based on current conditions"""
        recommendations = []
        
        if safety_level == 'critical':
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Crowd density at critical levels",
                "Stop entry of new people immediately",
                "Activate emergency crowd control procedures",
                "Consider evacuation of high-density areas"
            ])
        elif safety_level == 'high_risk':
            recommendations.extend([
                "High crowd density detected - monitor closely",
                "Reduce entry rate or temporarily stop new entries",
                "Deploy additional staff to high-density areas",
                "Prepare emergency procedures"
            ])
        elif safety_level == 'caution':
            recommendations.extend([
                "Moderate crowd density - increased monitoring recommended",
                "Consider crowd flow management measures",
                "Alert security staff to monitor situation"
            ])
        
        if len(hotspots) > 0:
            recommendations.append(f"Address {len(hotspots)} density hotspot(s) detected")
        
        if trend == 'increasing':
            recommendations.append("Density is increasing - proactive measures recommended")
        
        return recommendations

class CrowdDensityAnalyzer:
    """Main class for comprehensive crowd density analysis"""
    
    def __init__(self, frame_shape=None):
        self.density_calculator = DensityCalculator(frame_shape)
        self.historical_data = []
        self.alert_thresholds = {
            'caution': 10.0,
            'warning': 15.0,
            'critical': 25.0
        }
    
    def analyze_crowd_density(self, detections, frame_shape=None, method='grid_based'):
        """Comprehensive crowd density analysis"""
        # Calculate density using specified method
        density_result = self.density_calculator.calculate_density(
            detections, frame_shape, method
        )
        
        # Safety assessment
        safety_assessment = self.density_calculator.get_safety_assessment(density_result)
        
        # Combine results
        analysis_result = {
            'timestamp': time.time(),
            'density_analysis': density_result,
            'safety_assessment': safety_assessment,
            'person_count': len(detections),
            'analysis_method': method
        }
        
        # Store historical data
        self.historical_data.append(analysis_result)
        if len(self.historical_data) > 1000:  # Keep last 1000 analyses
            self.historical_data.pop(0)
        
        return analysis_result
    
    def get_density_trends(self, time_window=300):  # 5 minutes default
        """Get density trends over specified time window"""
        current_time = time.time()
        recent_data = [
            data for data in self.historical_data
            if current_time - data['timestamp'] <= time_window
        ]
        
        if len(recent_data) < 2:
            return {
                'trend': 'insufficient_data',
                'trend_direction': 'stable',
                'peak_density': 0,
                'avg_density': 0
            }
        
        densities = [data['density_analysis']['overall_density'] for data in recent_data]
        timestamps = [data['timestamp'] for data in recent_data]
        
        # Calculate trend using linear regression
        if len(densities) >= 3:
            x = np.array(timestamps)
            y = np.array(densities)
            
            # Normalize x for better numerical stability
            x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.ones_like(x)
            
            # Simple linear regression
            slope = np.corrcoef(x_norm, y)[0, 1] * np.std(y) / np.std(x_norm)
            
            if abs(slope) < 0.1:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
            slope = 0
        
        return {
            'trend': 'calculated',
            'trend_direction': trend_direction,
            'trend_slope': float(slope),
            'peak_density': float(max(densities)),
            'avg_density': float(np.mean(densities)),
            'min_density': float(min(densities)),
            'data_points': len(recent_data),
            'time_span': float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        }