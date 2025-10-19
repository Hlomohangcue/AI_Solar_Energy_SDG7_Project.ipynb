# Data Directory

This directory contains datasets used in the Solar Energy Prediction project.

## Files Description:

### Raw Data
- `solar_weather_data.csv` - Generated synthetic weather and solar irradiance dataset
- `weather_features.csv` - Processed weather features for training

### Processed Data
- `train_data.csv` - Training dataset (80% split)
- `test_data.csv` - Test dataset (20% split)
- `scaled_features.csv` - Standardized features for linear models

## Data Schema:

| Column | Type | Description | Range |
|--------|------|-------------|--------|
| `timestamp` | datetime | Date and time of measurement | 2023-01-01 to 2023-12-31 |
| `hour` | int | Hour of day | 0-23 |
| `month` | int | Month of year | 1-12 |
| `temperature` | float | Air temperature (°C) | -5 to 40 |
| `humidity` | float | Relative humidity (%) | 10-100 |
| `cloud_cover` | float | Cloud coverage (%) | 0-100 |
| `wind_speed` | float | Wind speed (m/s) | 0-20 |
| `pressure` | float | Atmospheric pressure (hPa) | 980-1040 |
| `solar_elevation_proxy` | float | Solar position indicator | 0-1 |
| `solar_irradiance` | float | **Target**: Solar energy potential (W/m²) | 0-1200 |

## Engineered Features:
- `is_daytime` - Binary indicator for daylight hours (6 AM - 8 PM)
- `season` - Categorical season indicator (0=Winter, 1=Spring, 2=Summer, 3=Fall)
- `hour_sin`, `hour_cos` - Cyclical encoding of hour
- `month_sin`, `month_cos` - Cyclical encoding of month
- `temp_humidity_interaction` - Feature interaction term

## Usage:
```python
import pandas as pd

# Load the main dataset
df = pd.read_csv('data/solar_weather_data.csv')

# Load processed splits
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
```