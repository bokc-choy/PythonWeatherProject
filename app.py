from flask import Flask, render_template, request
import requests
from config import Config
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config.from_object(Config)

def get_weather_data(location):
    """Get current weather from WeatherAPI.com"""
    try:
        url = f"{app.config['WEATHERAPI_BASE_URL']}/forecast.json?key={app.config['WEATHERAPI_KEY']}&q={location}&days=3&aqi=no&alerts=no"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def create_forecast_plot(forecast_data):
    """Create temperature forecast plot"""
    try:
        #plot data
        fig, ax = plt.subplots(figsize=(10, 5))
        times = []
        temps = []
        
        # 24 hr forecast
        for hour in forecast_data['forecast']['forecastday'][0]['hour'] + \
                 forecast_data['forecast']['forecastday'][1]['hour'][:4]:
            times.append(datetime.strptime(hour['time'], '%Y-%m-%d %H:%M'))
            temps.append(hour['temp_c'])

        ax.plot(times, temps, marker='o', label='Temperature (°C)')
        ax.set_title('24-Hour Temperature Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True)
        plt.xticks(rotation=45)
        fig.tight_layout()

        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close(fig)
        
        return base64.b64encode(img_bytes.read()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form.get('location', 'London')
        weather_data = get_weather_data(location)
        
        if weather_data:
            plot = create_forecast_plot(weather_data)
            current_temp = weather_data['current']['temp_c']
            conditions = weather_data['current']['condition']['text']
            
            #display
            formatted_hours = []
            for hour in weather_data['forecast']['forecastday'][0]['hour'][:12]:
                hour_time = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M')
                formatted_hours.append({
                    'time': hour_time.strftime('%H:%M'),
                    'temp_c': hour['temp_c'],
                    'condition': hour['condition']['text'],
                    'precip_mm': hour['precip_mm'],
                    'humidity': hour['humidity']
                })
            
            return render_template('results.html', 
                                plot=plot,
                                location=location,
                                current_temp=current_temp,
                                conditions=conditions,
                                hours=formatted_hours)
        
        return render_template('index.html', error="Failed to fetch weather data")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)