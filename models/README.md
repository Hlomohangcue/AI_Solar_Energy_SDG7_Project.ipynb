# Models Directory

This directory contains trained machine learning models and related files for the Solar Energy Prediction project.

## Model Files:

### Trained Models
- `solar_energy_model.pkl` - Best performing Random Forest model (tuned)
- `linear_regression_model.pkl` - Baseline Linear Regression model
- `gradient_boosting_model.pkl` - Gradient Boosting Regressor model
- `feature_scaler.pkl` - StandardScaler for feature normalization

### Model Artifacts
- `hyperparameters.json` - Optimized hyperparameters from Grid Search
- `feature_names.json` - List of feature names used in training
- `model_metadata.json` - Model training metadata and performance metrics

## Model Performance Summary:

| Model | MAE (W/m²) | RMSE (W/m²) | R² Score | Training Time |
|-------|------------|-------------|----------|---------------|
| Linear Regression | 37.82 | 50.15 | 0.9138 | ~1s |
| Random Forest | 22.89 | 31.02 | 0.9546 | ~15s |
| Gradient Boosting | 24.12 | 33.45 | 0.9475 | ~25s |
| **Random Forest (Tuned)** | **21.73** | **29.45** | **0.9570** | ~45s |

## Usage:

### Loading a Saved Model
```python
import joblib
import numpy as np

# Load the best model and scaler
model = joblib.load('models/solar_energy_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Make predictions
def predict_solar_energy(features):
    # features should be a 2D array with 13 columns
    scaled_features = scaler.transform(features)
    prediction = model.predict(features)  # Note: RF doesn't need scaling
    return prediction
```

### Saving a New Model
```python
import joblib

# Save model
joblib.dump(trained_model, 'models/new_model.pkl')
joblib.dump(scaler, 'models/new_scaler.pkl')
```

## Model Details:

### Random Forest (Best Model)
- **Algorithm**: Random Forest Regressor
- **Trees**: 200 estimators
- **Max Depth**: 20
- **Min Samples Split**: 2
- **Min Samples Leaf**: 1
- **Features**: 13 engineered features
- **Cross-Validation**: 5-fold CV
- **Optimization**: Grid Search

### Feature Importance (Top 5):
1. `solar_elevation_proxy` (0.598) - Sun position
2. `cloud_cover` (0.186) - Weather obstruction
3. `temperature` (0.088) - Thermal correlation
4. `is_daytime` (0.045) - Day/night indicator
5. `season` (0.032) - Seasonal patterns