import requests
import numpy as np
from typing import Optional, Dict, Any, Tuple
from flask import current_app
from algorithms import CustomTempPredictor, custom_clustering, detect_anomalies

class WeatherDataProcessor:
    """Handles data collection, processing, and ML integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.weatherapi.com/v1"
    
    def get_processed_weather_data(self, location: str) -> Optional[Dict[str, Any]]:
        """Get and process weather data with ML predictions"""
        try:
            # Get raw data from API
            raw_data = self._fetch_api_data(location)
            if not raw_data:
                return None
                
            # Process basic weather data
            processed = self._process_basic_data(raw_data)
            
            # Apply ML algorithms
            processed.update(self._apply_ml_algorithms(raw_data))
            
            return processed
            
        except Exception as e:
            current_app.logger.error(f"Data processing error: {e}")
            return None
    
    def _fetch_api_data(self, location: str) -> Optional[Dict]:
        """Make API call and return raw response"""
        try:
            url = f"{self.base_url}/forecast.json?key={self.api_key}&q={location}&days=3"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"API request failed: {e}")
            return None
    
    def _process_basic_data(self, raw_data: Dict) -> Dict[str, Any]:
        """Process basic weather information"""
        return {
            'location': raw_data['location']['name'],
            'current_temp': raw_data['current']['temp_c'],
            'conditions': raw_data['current']['condition']['text'],
            'forecast_hours': [
                {
                    'time': hour['time'].split()[1],
                    'temp_c': hour['temp_c'],
                    'humidity': hour['humidity'],
                    'precip_mm': hour['precip_mm'],
                    'condition': hour['condition']['text']
                }
                for hour in raw_data['forecast']['forecastday'][0]['hour']
            ]
        }
    
    def _apply_ml_algorithms(self, raw_data: Dict) -> Dict[str, Any]:
        """Apply custom ML algorithms to the data"""
        # Prepare data for ML
        hours_data = raw_data['forecast']['forecastday'][0]['hour']
        temps = np.array([hour['temp_c'] for hour in hours_data])
        clustering_data = np.array([[hour['temp_c'], hour['humidity']] for hour in hours_data])
        
        # Apply algorithms
        anomalies = detect_anomalies(temps, window_size=3, threshold=2.0)
        cluster_labels, _ = custom_clustering(clustering_data, n_clusters=2)  # Unpack the tuple
        
        return {
            'anomalies': anomalies.tolist(),
            'clusters': cluster_labels.tolist(),  # Use just the labels part
            'next_temp_prediction': float(self._predict_next_temp(temps))
        }

    def _predict_next_temp(self, temps: np.ndarray) -> float:
        """Helper method for temperature prediction"""
        X = np.array([[i] for i in range(len(temps))])
        y = temps
        predictor = CustomTempPredictor(learning_rate=0.01, n_iterations=1000)
        predictor.fit(X, y)
        return predictor.predict(np.array([[len(temps)]]))[0]

    def find_similar_climate_regions(self, location: str, num_regions: int = 5) -> Optional[Dict[str, Any]]:
        """Find regions with similar climate patterns using clustering"""
        try:
            # Get the target location's climate data
            target_data = self._fetch_api_data(location)
            if not target_data:
                return None
                
            # Prepare comparison regions
            comparison_regions = [
                'London', 'New York', 'Tokyo', 'Sydney', 'Paris',
                'Berlin', 'Moscow', 'Dubai', 'Toronto', 'Los Angeles',
                'Singapore', 'Beijing', 'Rio de Janeiro', 'Cairo', 'Mumbai'
            ]
            
            # Collect climate features for all regions
            all_features = []
            region_names = []
            
            # Add target location first
            target_hours = target_data['forecast']['forecastday'][0]['hour']
            target_features = self._extract_daily_features(target_hours)
            all_features.append(target_features)
            region_names.append(location)
            
            # Add comparison regions
            for region in comparison_regions:
                if region.lower() == location.lower():
                    continue
                    
                data = self._fetch_api_data(region)
                if data:
                    hours = data['forecast']['forecastday'][0]['hour']
                    features = self._extract_daily_features(hours)
                    all_features.append(features)
                    region_names.append(region)
            
            if len(all_features) < 3:  # Need at least a few regions to cluster
                return None
                
            # Convert to numpy array
            feature_matrix = np.array(all_features)
            
            # Apply clustering (using 3 clusters)
            cluster_labels, cluster_centers = custom_clustering(feature_matrix, n_clusters=3)
            
            temp_means = [center[0] for center in cluster_centers]  # Index 0 is mean temp
            cluster_order = np.argsort(temp_means)  # Coldest to hottest
            remapped_labels = np.zeros_like(cluster_labels)
            
            for new_id, old_id in enumerate(cluster_order):
                remapped_labels[cluster_labels == old_id] = new_id
            target_cluster = cluster_labels[0]
            
            # Find other regions in the same cluster
            similar_regions = []
            for i, (name, label) in enumerate(zip(region_names, cluster_labels)):
                if label == target_cluster and name.lower() != location.lower():
                    # Calculate distance to target for sorting
                    distance = np.linalg.norm(feature_matrix[i] - target_features)
                    similar_regions.append((name, distance))
            
            # Sort by distance and get top matches
            similar_regions.sort(key=lambda x: x[1])
            top_matches = [region[0] for region in similar_regions[:num_regions]]
            
            return {
                'similar_regions': top_matches,
                'target_location': location,
                'cluster_id': int(target_cluster)
            }
            
        except Exception as e:
            current_app.logger.error(f"Error finding similar regions: {e}")
            return None
    
    def _extract_daily_features(self, hourly_data: list) -> np.ndarray:
        """Extract important daily climate features from hourly data"""
        temps = np.array([hour['temp_c'] for hour in hourly_data])
        humidities = np.array([hour['humidity'] for hour in hourly_data])
        precipitations = np.array([hour['precip_mm'] for hour in hourly_data])
        
        return np.array([
            np.mean(temps),        # Mean temperature
            np.std(temps),         # Temperature variability
            np.max(temps) - np.min(temps),  # Temperature range
            np.mean(humidities),   # Mean humidity
            np.sum(precipitations)  # Total precipitation
        ])