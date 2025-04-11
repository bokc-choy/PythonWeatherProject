import unittest
from unittest.mock import patch, MagicMock
from app import app, create_forecast_plot
import json
import os
import base64
import io

class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the specific sample data
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project_root
        data_path = os.path.join(base_dir, "data", "sample.json")  # Adjusted path
        with open(data_path, 'r') as f:
            cls.sample_data = json.load(f)
        
        # Process the sample data into forecast hours format
        cls.sample_forecast_hours = [
            {
                'time': hour['time'].split()[1],  # Extract just the time part
                'temp_c': hour['temp_c'],
                'humidity': hour['humidity'],
                'precip_mm': hour['precip_mm'],
                'condition': hour['condition']['text']
            }
            for hour in cls.sample_data['forecast']['forecastday'][0]['hour']
        ]
        
        # Expected values from the sample data
        cls.expected_location = "TestCity"
        cls.expected_current_temp = 20
        cls.expected_conditions = "Sunny"
        
        # Create test client
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_create_forecast_plot(self):
        """
        Test the plotting function with the sample data
        """
        plot_data = create_forecast_plot(self.sample_forecast_hours)
        
        # Basic validation
        self.assertIsNotNone(plot_data, "Plot data should not be None")
        self.assertIsInstance(plot_data, str, "Plot data should be a string")
        
        # Verify it's a valid base64 encoded PNG image
        try:
            decoded = base64.b64decode(plot_data)
            img = io.BytesIO(decoded)
            self.assertEqual(img.read(4), b'\x89PNG', "Should be a valid PNG image")
        except Exception as e:
            self.fail(f"Plot data is not valid base64 encoded PNG: {str(e)}")

    def test_plot_data_handling(self):
        """
        Test the plot function handles the sample data correctly
        """
        self.assertEqual(len(self.sample_forecast_hours), 5, "Should have 5 hours of data")
        self.assertEqual(self.sample_forecast_hours[0]['time'], "00:00")
        self.assertEqual(self.sample_forecast_hours[0]['temp_c'], 0)
        self.assertEqual(self.sample_forecast_hours[-1]['time'], "04:00")
        self.assertEqual(self.sample_forecast_hours[-1]['temp_c'], 3)

    def test_index_route_get(self):
        """
        Test the home page loads correctly
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200, "Should return status 200")
        self.assertIn(b"Weather Prediction App", response.data, "Should show app title")

    @patch('app.weather_processor')
    def test_index_route_post_success(self, mock_processor):
        """
        Test successful weather data submission
        """
        mock_processor.get_processed_weather_data.return_value = {
            'location': self.expected_location,
            'current_temp': self.expected_current_temp,
            'conditions': self.expected_conditions,
            'forecast_hours': self.sample_forecast_hours,
            'anomalies': [False] * len(self.sample_forecast_hours),
            'clusters': [0] * len(self.sample_forecast_hours),
            'next_temp_prediction': 4.0  # Next predicted temp
        }

        # Mock the plot function to return a test string
        with patch('app.create_forecast_plot', return_value="test_plot_data"):
            response = self.client.post('/', data={'location': 'TestCity'})
            
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"TestCity", response.data)
            self.assertIn(b"20", response.data)     # Current temp
            self.assertIn(b"Sunny", response.data)  # Conditions
            
            # Verify mock was called correctly
            mock_processor.get_processed_weather_data.assert_called_once_with('TestCity')

    @patch('app.weather_processor')
    def test_index_route_post_failure(self, mock_processor):
        """
        Test failed weather data fetch
        """
        mock_processor.get_processed_weather_data.return_value = None
        
        response = self.client.post('/', data={'location': 'InvalidLocation'})
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Failed to fetch weather data", response.data)

if __name__ == '__main__':
    unittest.main()