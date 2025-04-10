import requests
import numpy as np
from typing import Optional, Dict, Any, Tuple
from flask import current_app
from algorithms import CustomTempPredictor, custom_cluserting, detect_anomalies

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
        clusters = custom_cluserting(clustering_data, n_clusters=2)
        
        # Temperature prediction
        X = np.array([[i] for i in range(len(temps))])
        y = temps
        predictor = CustomTempPredictor(learning_rate=0.01, n_iterations=1000)
        predictor.fit(X, y)
        next_temp = predictor.predict(np.array([[len(temps)]]))[0]
        
        return {
            'anomalies': anomalies.tolist(),
            'clusters': clusters.tolist(),
            'next_temp_prediction': float(next_temp)
        }