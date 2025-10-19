from flask import Flask, render_template_string, request
import joblib
import numpy as np
import os
import sklearn

# Load model and scaler
MODEL_PATH = os.path.join('models', 'solar_energy_model.pkl')
SCALER_PATH = os.path.join('models', 'feature_scaler.pkl')
FEATURES_PATH = os.path.join('models', 'feature_names.json')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature engineering function (same as notebook)
def get_features(temperature, humidity, cloud_cover, wind_speed, pressure, hour, month):
    hour_factor = np.sin((hour - 6) * np.pi / 12)
    hour_factor = max(hour_factor, 0)
    day_of_year = (month - 1) * 30 + 15
    season_factor = np.sin((day_of_year - 80) * 2 * np.pi / 365)
    season_factor = 0.5 + 0.5 * season_factor
    solar_elevation_proxy = hour_factor * season_factor
    is_daytime = 1 if 6 <= hour <= 20 else 0
    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    temp_humidity_interaction = temperature * humidity / 100
    return np.array([[
        temperature, humidity, cloud_cover, wind_speed, pressure,
        solar_elevation_proxy, is_daytime, season,
        hour_sin, hour_cos, month_sin, month_cos,
        temp_humidity_interaction
    ]])

app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <title>Solar Energy Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; }
        .container { max-width: 500px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        h2 { color: #2c3e50; }
        label { display: block; margin-top: 15px; }
        input[type=number] { width: 100%; padding: 8px; margin-top: 5px; }
        button { margin-top: 20px; padding: 10px 20px; background: #27ae60; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
        .result { background: #eafaf1; padding: 15px; border-radius: 6px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class='container'>
        <h2>Solar Energy Potential Prediction</h2>
        <form method='post'>
            <label>Temperature (°C): <input type='number' name='temperature' step='0.1' required></label>
            <label>Humidity (%): <input type='number' name='humidity' step='0.1' required></label>
            <label>Cloud Cover (%): <input type='number' name='cloud_cover' step='0.1' required></label>
            <label>Wind Speed (m/s): <input type='number' name='wind_speed' step='0.1' required></label>
            <label>Pressure (hPa): <input type='number' name='pressure' step='0.1' required></label>
            <label>Hour (0-23): <input type='number' name='hour' min='0' max='23' required></label>
            <label>Month (1-12): <input type='number' name='month' min='1' max='12' required></label>
            <button type='submit'>Predict</button>
        </form>
        {% if result %}
        <div class='result'>
            <h3>Prediction Result</h3>
            <p><b>Solar Irradiance:</b> {{ result['irradiance'] }} W/m²</p>
            <p><b>Estimated Hourly Energy:</b> {{ result['energy_kwh'] }} kWh</p>
            <p><b>Season:</b> {{ result['season'] }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Version check utility
def check_sklearn_version(model_dir='models'):
    version_file = os.path.join(model_dir, 'scikit_learn_version.txt')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    model_version = line.strip()
                    break
            else:
                model_version = None
        current_version = sklearn.__version__
        if model_version and model_version != current_version:
            print(f"WARNING: Model was trained with scikit-learn {model_version}, but you are using {current_version}.")
            print("This may cause errors or unpredictable results. Consider retraining or using the same version.")
        else:
            print(f"scikit-learn version matches: {current_version}")
    else:
        print("No scikit_learn_version.txt found. Cannot check version compatibility.")

# Run version check at startup
check_sklearn_version()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            cloud_cover = float(request.form['cloud_cover'])
            wind_speed = float(request.form['wind_speed'])
            pressure = float(request.form['pressure'])
            hour = int(request.form['hour'])
            month = int(request.form['month'])
            features = get_features(temperature, humidity, cloud_cover, wind_speed, pressure, hour, month)
            prediction = model.predict(features)[0]
            prediction = max(0, prediction)
            panel_efficiency = 0.20
            panel_area = 1.6
            energy_kwh = round((prediction * panel_efficiency * panel_area) / 1000, 3)
            season_names = ['Winter', 'Spring', 'Summer', 'Fall']
            if month in [12, 1, 2]:
                season = season_names[0]
            elif month in [3, 4, 5]:
                season = season_names[1]
            elif month in [6, 7, 8]:
                season = season_names[2]
            else:
                season = season_names[3]
            result = {
                'irradiance': round(prediction, 2),
                'energy_kwh': energy_kwh,
                'season': season
            }
        except Exception as e:
            result = {'irradiance': 'Error', 'energy_kwh': str(e), 'season': 'N/A'}
    return render_template_string(HTML_FORM, result=result)

if __name__ == '__main__':
    app.run(debug=True)
