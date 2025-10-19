"""
Solar Energy Prediction Model Utilities

This module provides utility functions for loading and using the trained
solar energy prediction models.

Usage:
    from model_utils import load_model, predict_solar_energy
    
    # Load the model
    model_components = load_model()
    
    # Make prediction
    result = predict_solar_energy(model_components, 25, 60, 20, 4, 1013, 13, 7)
"""

import joblib
import json
import numpy as np
import os


def load_model(model_dir='models'):
    """
    Load the trained solar energy prediction model and components.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the model files
        
    Returns:
    --------
    dict or None: Dictionary with model components or None if error
    """
    try:
        # Construct file paths
        model_path = os.path.join(model_dir, 'solar_energy_model.pkl')
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        features_path = os.path.join(model_dir, 'feature_names.json')
        
        # Load components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'metadata': metadata,
            'feature_names': feature_names
        }
        
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please ensure all model files are in the correct directory.")
        return None
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return None


def calculate_derived_features(hour, month):
    """
    Calculate derived features from hour and month.
    
    Parameters:
    -----------
    hour : int
        Hour of day (0-23)
    month : int
        Month (1-12)
        
    Returns:
    --------
    dict: Dictionary of derived features
    """
    # Solar elevation proxy
    hour_factor = np.sin((hour - 6) * np.pi / 12)
    hour_factor = max(hour_factor, 0)
    
    day_of_year = (month - 1) * 30 + 15  # Approximate
    season_factor = np.sin((day_of_year - 80) * 2 * np.pi / 365)
    season_factor = 0.5 + 0.5 * season_factor
    
    solar_elevation_proxy = hour_factor * season_factor
    
    # Other derived features
    is_daytime = 1 if 6 <= hour <= 20 else 0
    
    if month in [12, 1, 2]:
        season = 0  # Winter
    elif month in [3, 4, 5]:
        season = 1  # Spring
    elif month in [6, 7, 8]:
        season = 2  # Summer
    else:
        season = 3  # Fall
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return {
        'solar_elevation_proxy': solar_elevation_proxy,
        'is_daytime': is_daytime,
        'season': season,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos
    }


def predict_solar_energy(model_components, temperature, humidity, cloud_cover, 
                        wind_speed, pressure, hour, month):
    """
    Predict solar energy irradiance using the trained model.
    
    Parameters:
    -----------
    model_components : dict
        Dictionary returned by load_model()
    temperature : float
        Temperature in Celsius
    humidity : float
        Humidity percentage (0-100)
    cloud_cover : float
        Cloud cover percentage (0-100)
    wind_speed : float
        Wind speed in m/s
    pressure : float
        Atmospheric pressure in hPa
    hour : int
        Hour of day (0-23)
    month : int
        Month (1-12)
        
    Returns:
    --------
    dict: Prediction results and metadata
    """
    if model_components is None:
        return {"error": "Model not loaded properly"}
    
    # Extract model components
    model = model_components['model']
    
    # Calculate derived features
    derived = calculate_derived_features(hour, month)
    
    # Temperature-humidity interaction
    temp_humidity_interaction = temperature * humidity / 100
    
    # Create feature array (must match training order)
    features = np.array([[
        temperature, humidity, cloud_cover, wind_speed, pressure,
        derived['solar_elevation_proxy'], derived['is_daytime'], derived['season'],
        derived['hour_sin'], derived['hour_cos'], derived['month_sin'], derived['month_cos'],
        temp_humidity_interaction
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    prediction = max(0, prediction)  # Ensure non-negative
    
    # Calculate additional metrics
    panel_efficiency = 0.20  # 20%
    panel_area = 1.6  # m²
    hourly_energy_kwh = (prediction * panel_efficiency * panel_area) / 1000
    daily_estimate = hourly_energy_kwh * 24 * derived['solar_elevation_proxy']
    
    return {
        'solar_irradiance_w_per_m2': round(prediction, 2),
        'hourly_energy_kwh': round(hourly_energy_kwh, 4),
        'daily_estimate_kwh': round(daily_estimate, 2),
        'input_conditions': {
            'temperature': temperature,
            'humidity': humidity,
            'cloud_cover': cloud_cover,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'hour': hour,
            'month': month
        },
        'derived_features': {
            'solar_elevation': round(derived['solar_elevation_proxy'], 3),
            'is_daytime': bool(derived['is_daytime']),
            'season': ['Winter', 'Spring', 'Summer', 'Fall'][derived['season']]
        }
    }


def batch_predict(model_components, weather_data):
    """
    Make predictions for multiple weather conditions.
    
    Parameters:
    -----------
    model_components : dict
        Dictionary returned by load_model()
    weather_data : list of dict
        List of weather condition dictionaries
        
    Returns:
    --------
    list: List of prediction results
    """
    predictions = []
    
    for conditions in weather_data:
        result = predict_solar_energy(
            model_components,
            conditions['temperature'],
            conditions['humidity'],
            conditions['cloud_cover'],
            conditions['wind_speed'],
            conditions['pressure'],
            conditions['hour'],
            conditions['month']
        )
        predictions.append(result)
    
    return predictions


# Example usage
if __name__ == "__main__":
    # Load model
    model_components = load_model()
    
    if model_components:
        # Test prediction
        result = predict_solar_energy(
            model_components, 
            temperature=25, humidity=60, cloud_cover=20,
            wind_speed=4, pressure=1013, hour=13, month=7
        )
        
        print("Solar Energy Prediction:")
        print(f"Irradiance: {result['solar_irradiance_w_per_m2']} W/m²")
        print(f"Hourly Energy: {result['hourly_energy_kwh']} kWh")
        print(f"Season: {result['derived_features']['season']}")