import os
os.environ["CODECARBON_FORCE_MODE_CPU_LOAD"] = "True"
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from codecarbon import EmissionsTracker

feature_columns = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean"]
def get_city_data(latitude, longitude): 
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "daily": feature_columns,
        "timezone": "America/Los_Angeles"
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["next_day_temp_max"] = df["temperature_2m_max"].shift(-1)
    df = df.dropna()
    return df
def run_models_for_city(city_name, latitude, longitude):
    print(f"\n=========== {city_name} ===========")
    df = get_city_data(latitude, longitude)

    X = df[feature_columns]
    y = df["next_day_temp_max"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tracker_1 = EmissionsTracker(measure_power_secs=15)
    tracker_1.start()
    model_1 = LinearRegression()
    model_1.fit(X_train, y_train)
    emissions_1 = tracker_1.stop()
    predictions_1 = model_1.predict(X_test)
    mae_1 = mean_absolute_error(y_test, predictions_1)
    r2_1 = r2_score(y_test, predictions_1)

    print(f"Linear Regression -> MAE: {mae_1:.2f}, R2: {r2_1:.3f}, Emissions: {emissions_1:.10f}")

    tracker_2 = EmissionsTracker(measure_power_secs=15)
    tracker_2.start()
    model_2 = RandomForestRegressor(random_state=42)
    model_2.fit(X_train, y_train)
    emissions_2 = tracker_2.stop()
    predictions_2 = model_2.predict(X_test)
    mae_2 = mean_absolute_error(y_test, predictions_2)
    r2_2 = r2_score(y_test, predictions_2)

    print(f"Random Forest     -> MAE: {mae_2:.2f}, R2: {r2_2:.3f}, Emissions: {emissions_2:.10f}")

    print("Feature importance (Random Forest):")
    for name, importance in zip(feature_columns, model_2.feature_importances_):
        print(f" {name}: {importance:.4f}")
    
    return {
        "city": city_name,
        "lr_mae": mae_1, "lr_r2": r2_1, "lr_emissions": emissions_1,
        "rf_mae": mae_2, "rf_r2": r2_2, "rf_emissions": emissions_2,
    }

cities = [
    ("Sacramento", 38.5816, -121.4944),
    ("Denver", 39.7392, -104.9903),
    ("Miami", 25.7617, -80.1918),
]

all_results = []
for city_name, lat, lon in cities:
    result = run_models_for_city(city_name, lat, lon)
    all_results.append(result)

# Save results so you can build your chart from this later
results_df = pd.DataFrame(all_results)
results_df.to_csv("multi_city_results.csv", index=False)
print("\nSaved all results to multi_city_results.csv")
