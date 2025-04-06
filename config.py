import os

class Config:
    # WeatherAPI
    WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY', 'c28c0f8792a2458788221356250604')
    WEATHERAPI_BASE_URL = "http://api.weatherapi.com/v1"