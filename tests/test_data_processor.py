import unittest
from unittest.mock import patch, MagicMock
from data_processor import WeatherDataProcessor
import json
import os

class TestWeatherDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = WeatherDataProcessor(api_key="dummy_key")
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project_root
        data_path = os.path.join(base_dir, "data", "sample.json")  # Adjusted path
        with open(data_path, "r") as f:
            self.mock_raw_data = json.load(f)
    
    def tearDown(self):
        self.processor = None
        self.mock_raw_data = None

    @patch('data_processor.requests.get')
    def test_fetch_api_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"location": {"name": "TestCity"}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = self.processor._fetch_api_data("TestCity")
        self.assertIn("location", result)
        mock_get.assert_called_once()

    def test_process_basic_data(self):
        # Use the mock data loaded in setUp() for processing
        result = self.processor._process_basic_data(self.mock_raw_data)
        self.assertEqual(result["location"], "TestCity")  # Assuming this is part of your sample data
        self.assertEqual(result["current_temp"], 20)  # Adjust according to your mock data
        self.assertGreater(len(result["forecast_hours"]), 0)  # Assuming there's forecast data

    def test_apply_ml_algorithms(self):
        # Use the mock data loaded in setUp() for ML processing
        result = self.processor._apply_ml_algorithms(self.mock_raw_data)
        self.assertIn("anomalies", result)
        self.assertIn("clusters", result)
        self.assertIn("next_temp_prediction", result)
        self.assertEqual(len(result["anomalies"]), 5)  # Assuming there are 24 hours of forecast data
        self.assertEqual(len(result["clusters"]), 5)

    @patch.object(WeatherDataProcessor, '_fetch_api_data')
    @patch.object(WeatherDataProcessor, '_process_basic_data')
    @patch.object(WeatherDataProcessor, '_apply_ml_algorithms')
    def test_get_processed_weather_data(self, mock_apply_ml, mock_process_basic, mock_fetch_api):
        mock_fetch_api.return_value = self.mock_raw_data
        mock_process_basic.return_value = {"location": "TestCity"}
        mock_apply_ml.return_value = {"anomalies": [False], "clusters": [1], "next_temp_prediction": 25.5}

        result = self.processor.get_processed_weather_data("TestCity")
        self.assertEqual(result["location"], "TestCity")
        self.assertEqual(result["clusters"], [1])
        self.assertEqual(result["next_temp_prediction"], 25.5)

if __name__ == "__main__":
    unittest.main()
