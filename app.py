from flask import Flask, render_template, request
from data_processor import WeatherDataProcessor
from config import Config
import matplotlib
import matplotlib.dates as mdates
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from datetime import datetime, time

CLUSTER_DESCRIPTIONS = {
    0: "Hot and Dry Climate",
    1: "Temperate and Humid Climate", 
    2: "Cold and Variable Climate",
    'N/A': "Climate classification not available"
}

app = Flask(__name__)
app.config.from_object(Config)

# Initialize the WeatherDataProcessor
weather_processor = WeatherDataProcessor(api_key=app.config['WEATHERAPI_KEY'])

def convert_to_f(c):
    return ((c*9/5)+32)

def create_forecast_plot(forecast_hours):
    """Create temperature forecast plot from processed hourly data"""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        times = []
        temps = []
        
        #Convert string times to datetime objects
        today = datetime.now().date()
        for hour in forecast_hours[:24]:  # Use first 24 hours
            hour_str = hour['time']
            # Parse "HH:MM" format
            hour_part, minute_part = map(int, hour_str.split(':'))
            dt = datetime.combine(today, time(hour_part, minute_part))
            times.append(dt)
            temps.append(hour['temp_c'])

        #Formatting
        ax.plot(times, temps, marker='o', linestyle='-', color='#1f77b4', 
               label='Temperature (°C)')
        
        ax.set_title('24-Hour Temperature Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
        
        #x-axis (time)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png', bbox_inches='tight', dpi=100)
        img_bytes.seek(0)
        plt.close(fig)
        
        return base64.b64encode(img_bytes.read()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form.get('location', 'London')
        weather_data = weather_processor.get_processed_weather_data(location)
        
        if weather_data:
            plot = create_forecast_plot(weather_data['forecast_hours'])
            current_temp = weather_data['current_temp']
            current_temp_f = convert_to_f(current_temp)
            conditions = weather_data['conditions']
            
            # Initialize default values
            similar_regions = None
            cluster_id = 'N/A'
            
            # Get similar climate regions if weather data exists
            similar_regions_result = weather_processor.find_similar_climate_regions(location)
            if similar_regions_result:
                similar_regions = similar_regions_result.get('similar_regions')
                cluster_id = similar_regions_result.get('cluster_id', 'N/A')
            
            #Display processed data
            formatted_hours = []
            for i, hour in enumerate(weather_data['forecast_hours'][:12]):
                formatted_hours.append({
                    'time': hour['time'],
                    'temp_c': hour['temp_c'],
                    'condition': hour['condition'],
                    'precip_mm': hour['precip_mm'],
                    'humidity': hour['humidity'],
                    'anomaly': bool(weather_data['anomalies'][i]) if i < len(weather_data['anomalies']) else False,
                    'cluster': int(weather_data['clusters'][i])
                })
            
            return render_template('results.html', 
                                plot=plot,
                                location=location,
                                current_temp=current_temp,
                                current_temp_f=current_temp_f,
                                conditions=conditions,
                                hours=formatted_hours,
                                next_temp=round(weather_data['next_temp_prediction'], 2),
                                similar_regions=similar_regions,
                                cluster_id=cluster_id,
                                cluster_description=CLUSTER_DESCRIPTIONS.get(cluster_id, "Unknown climate type"))
        
        return render_template('index.html', error="Failed to fetch weather data")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)