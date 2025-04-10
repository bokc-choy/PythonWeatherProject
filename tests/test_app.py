import unittest
from unittest.mock import patch, MagicMock
from app import app, get_weather_data, create_forecast_plot
import json
import os

class TestApp(unittest.TestCase):
    
    def setUp(self):
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

        # Load the sample data from the JSON file in the 'data' directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, "data", "sample.json")  # Adjusted path
        with open(data_path, "r") as f:
            self.sample_data = json.load(f)

    def tearDown(self):
        self.app_context.pop()

    @patch('app.requests.get')
    def test_get_weather_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'location': {'name': 'London'}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        data = get_weather_data('London')
        self.assertIn('location', data)
        mock_get.assert_called_once()

    def test_create_forecast_plot_invalid_data(self):
        result = create_forecast_plot({})
        self.assertIsNone(result)

    @patch('app.get_weather_data')
    def test_index_post_success(self, mock_get_weather_data):
        # Use the loaded sample data instead of raw data
        mock_get_weather_data.return_value = {
            'current': {'temp_c': self.sample_data['current']['temp_c'], 'condition': self.sample_data['current']['condition']},
            'forecast': {
                'forecastday': self.sample_data['forecast']['forecastday']
            }
        }
        
        response = self.client.post('/', data={'location': 'London'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Temperature Forecast', response.data)

if __name__ == '__main__':
    unittest.main()
