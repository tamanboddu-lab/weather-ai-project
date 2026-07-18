import os
os.environ["CODECARBON_FORCE_MODE_CPU_LOAD"] = "True"
import requests
import pandas as pd


# Sacramento, CA coordinates (swap these for your chosen location)
latitude = 38.5816
longitude = -121.4944

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean"],
    "timezone": "America/Los_Angeles"
}

response = requests.get(url, params=params)
data = response.json()

# Convert to a DataFrame
df = pd.DataFrame(data["daily"])
print(df.head())

# Save it as a CSV so you don't need to re-fetch it every time
df.to_csv("weather_data.csv", index=False)
print("Saved to weather_data.csv")

# --- NEW CODE BELOW THIS LINE ---

# Create the target: tomorrow's max temperature
df["next_day_temp_max"] = df["temperature_2m_max"].shift(-1)

# Drop the last row since it won't have a "tomorrow" to predict
df = df.dropna()

print(df.head())

# Define inputs (X) and target (y)
feature_columns = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean"]

X = df[feature_columns]
y = df["next_day_temp_max"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from codecarbon import EmissionsTracker

# Start tracking energy/emissions for this model
tracker = EmissionsTracker(measure_power_secs=15)
tracker.start()

# Train the simple model
model_1 = LinearRegression()
model_1.fit(X_train, y_train)

# Stop tracking and get emissions
emissions_1 = tracker.stop()

# Make predictions on the test set
predictions_1 = model_1.predict(X_test)

# Evaluate accuracy
mae_1 = mean_absolute_error(y_test, predictions_1)
r2_1 = r2_score(y_test, predictions_1)

print("=== Model 1: Linear Regression ===")
print("Mean Absolute Error:", mae_1)
print("R² Score:", r2_1)
print("Emissions (kg CO2):", emissions_1)   