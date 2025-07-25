#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class WeatherPredictor:
    def __init__(self):
        # Initialize models for precipitation, wind speed, and wind direction
        self.precip_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.wind_speed_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.wind_direction_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for the model (wind and other factors for predictions)
        """
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['month'] = pd.to_datetime(df['time']).dt.month
        # Features: wind, humidity, pressure, cloud, hour, month, altitude (if available)
        feature_columns = [
            'hour', 'month',
            'relative_humidity_2m',
            'pressure_msl',
            'wind_speed_10m',
            'wind_gusts_10m',
            'wind_direction_10m',
            'cloudcover'
        ]
        if 'altitude' in df.columns:
            feature_columns.append('altitude')
        for col in feature_columns + ['precipitation', 'wind_speed_10m', 'wind_direction_10m']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        targets = {
            'precipitation': df['precipitation'],
            'wind_speed_2h': df['wind_speed_10m'],  # Proxy for 2h forecast
            'wind_direction_2h': df['wind_direction_10m']  # Proxy for 2h forecast
        }
        return df[feature_columns], targets

    def train(self, X: pd.DataFrame, targets: dict):
        """
        Train the weather prediction models for precipitation, wind speed, and direction
        """
        # Defensive: check for NaN
        if X.isnull().any().any() or any(targets[t].isnull().any() for t in targets):
            raise ValueError("Training data contains NaN values.")
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Train models
        self.precip_model.fit(X_scaled, targets['precipitation'])
        self.wind_speed_model.fit(X_scaled, targets['wind_speed_2h'])
        self.wind_direction_model.fit(X_scaled, targets['wind_direction_2h'])
        
    def evaluate(self, X: pd.DataFrame, targets: dict) -> dict:
        """
        Evaluate the model performance for all targets
        """
        # Defensive: check for NaN
        if X.isnull().any().any() or any(targets[t].isnull().any() for t in targets):
            raise ValueError("Evaluation data contains NaN values.")
        X_scaled = self.scaler.transform(X)
        results = {}
        # Evaluate precipitation
        y_pred_precip = self.precip_model.predict(X_scaled)
        mse_precip = mean_squared_error(targets['precipitation'], y_pred_precip)
        rmse_precip = np.sqrt(mse_precip)
        r2_precip = r2_score(targets['precipitation'], y_pred_precip)
        results['precipitation'] = {'RMSE': rmse_precip, 'R2': r2_precip}
        # Evaluate wind speed
        y_pred_speed = self.wind_speed_model.predict(X_scaled)
        mse_speed = mean_squared_error(targets['wind_speed_2h'], y_pred_speed)
        rmse_speed = np.sqrt(mse_speed)
        r2_speed = r2_score(targets['wind_speed_2h'], y_pred_speed)
        results['wind_speed_2h'] = {'RMSE': rmse_speed, 'R2': r2_speed}
        # Evaluate wind direction
        y_pred_dir = self.wind_direction_model.predict(X_scaled)
        mse_dir = mean_squared_error(targets['wind_direction_2h'], y_pred_dir)
        rmse_dir = np.sqrt(mse_dir)
        r2_dir = r2_score(targets['wind_direction_2h'], y_pred_dir)
        results['wind_direction_2h'] = {'RMSE': rmse_dir, 'R2': r2_dir}
        return results
    
    def save_model(self, model_dir: str):
        """
        Save the trained models and scaler
        """
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.precip_model, os.path.join(model_dir, 'precip_model.joblib'))
        joblib.dump(self.wind_speed_model, os.path.join(model_dir, 'wind_speed_model.joblib'))
        joblib.dump(self.wind_direction_model, os.path.join(model_dir, 'wind_direction_model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))

def main():
    # Load historical data
    data_file = "data/historical_weather.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run fetch_historical_data.py first.")
        return
    print("Loading historical weather data...")
    df = pd.read_csv(data_file)
    # Drop rows where target or features are NaN
    df = df.dropna(subset=['precipitation', 'relative_humidity_2m', 'pressure_msl', 'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'cloudcover'])
    # Initialize predictor
    predictor = WeatherPredictor()
    # Prepare features
    print("Preparing features...")
    try:
        X, targets = predictor.prepare_features(df)
    except Exception as e:
        print(f"Feature preparation error: {e}")
        return
    # Split data
    X_train, X_test, y_train_precip, y_test_precip = train_test_split(
        X, targets['precipitation'], test_size=0.2, random_state=42
    )
    y_train = {
        'precipitation': y_train_precip,
        'wind_speed_2h': targets['wind_speed_2h'].loc[y_train_precip.index],
        'wind_direction_2h': targets['wind_direction_2h'].loc[y_train_precip.index]
    }
    y_test = {
        'precipitation': y_test_precip,
        'wind_speed_2h': targets['wind_speed_2h'].loc[y_test_precip.index],
        'wind_direction_2h': targets['wind_direction_2h'].loc[y_test_precip.index]
    }
    # Train models
    print("Training models...")
    try:
        predictor.train(X_train, y_train)
    except Exception as e:
        print(f"Training error: {e}")
        return
    # Evaluate
    print("\nEvaluating model performance...")
    try:
        train_metrics = predictor.evaluate(X_train, y_train)
        test_metrics = predictor.evaluate(X_test, y_test)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return
    print("\nTraining Set Metrics:")
    print("Precipitation:")
    print(f"RMSE: {train_metrics['precipitation']['RMSE']:.2f} mm")
    print(f"R2 Score: {train_metrics['precipitation']['R2']:.3f}")
    print("Wind Speed (2h):")
    print(f"RMSE: {train_metrics['wind_speed_2h']['RMSE']:.2f} m/s")
    print(f"R2 Score: {train_metrics['wind_speed_2h']['R2']:.3f}")
    print("Wind Direction (2h):")
    print(f"RMSE: {train_metrics['wind_direction_2h']['RMSE']:.2f} degrees")
    print(f"R2 Score: {train_metrics['wind_direction_2h']['R2']:.3f}")
    print("\nTest Set Metrics:")
    print("Precipitation:")
    print(f"RMSE: {test_metrics['precipitation']['RMSE']:.2f} mm")
    print(f"R2 Score: {test_metrics['precipitation']['R2']:.3f}")
    print("Wind Speed (2h):")
    print(f"RMSE: {test_metrics['wind_speed_2h']['RMSE']:.2f} m/s")
    print(f"R2 Score: {test_metrics['wind_speed_2h']['R2']:.3f}")
    print("Wind Direction (2h):")
    print(f"RMSE: {test_metrics['wind_direction_2h']['RMSE']:.2f} degrees")
    print(f"R2 Score: {test_metrics['wind_direction_2h']['R2']:.3f}")
    # Save models
    print("\nSaving models...")
    predictor.save_model("models")
    print("Models saved in 'models' directory")

if __name__ == "__main__":
    main()