import os
os.environ["CODECARBON_FORCE_MODE_CPU_LOAD"] = "True"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from codecarbon import EmissionsTracker

# Reload the saved data
df = pd.read_csv("weather_data.csv")

# Recreate the target column
df["next_day_temp_max"] = df["temperature_2m_max"].shift(-1)
df = df.dropna()

# Recreate X and y
feature_columns = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean"]
X = df[feature_columns]
y = df["next_day_temp_max"]

# Recreate the same train/test split (same random_state = same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now build Model 2
tracker_2 = EmissionsTracker(measure_power_secs=15)
tracker_2.start()

model_2 = RandomForestRegressor(random_state=42)
model_2.fit(X_train, y_train)

emissions_2 = tracker_2.stop()

predictions_2 = model_2.predict(X_test)

mae_2 = mean_absolute_error(y_test, predictions_2)
r2_2 = r2_score(y_test, predictions_2)

print("=== Model 2: Random Forest ===")
print("Mean Absolute Error:", mae_2)
print("R² Score:", r2_2)
print("Emissions (kg CO2):", emissions_2)