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
        self.model = RandomForestRegressor(
            n_estimators=600,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=7,
            random_state=50
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for the model
        """
        # Extract hour from timestamp
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['month'] = pd.to_datetime(df['time']).dt.month

        # Features to use
        feature_columns = [
            'hour', 'month',
            'relative_humidity_2m',
            'pressure_msl',
            'wind_speed_10m',
            'wind_direction_10m',
            'cloud_cover'
        ]
        # Defensive: check for missing columns
        for col in feature_columns + ['temperature_2m']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        # Target variable (temperature)
        target = df['temperature_2m']
        return df[feature_columns], target

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the weather prediction model
        """
        # Defensive: check for NaN
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Training data contains NaN values.")
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Train model
        self.model.fit(X_scaled, y)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate the model performance
        """
        # Defensive: check for NaN
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Evaluation data contains NaN values.")
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        return {
            'RMSE': rmse,
            'R2': r2
        }
    
    def save_model(self, model_dir: str):
        """
        Save the trained model and scaler
        """
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'weather_model.joblib'))
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
    df = df.dropna(subset=['temperature_2m', 'relative_humidity_2m', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover'])
    # Initialize predictor
    predictor = WeatherPredictor()
    # Prepare features
    print("Preparing features...")
    try:
        X, y = predictor.prepare_features(df)
    except Exception as e:
        print(f"Feature preparation error: {e}")
        return
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train model
    print("Training model...")
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
    print(f"RMSE: {train_metrics['RMSE']:.2f}C")
    print(f"R2 Score: {train_metrics['R2']:.3f}")
    print("\nTest Set Metrics:")
    print(f"RMSE: {test_metrics['RMSE']:.2f}C")
    print(f"R2 Score: {test_metrics['R2']:.3f}")
    # Save model
    print("\nSaving model...")
    predictor.save_model("models")
    print("Model saved in 'models' directory")

if __name__ == "__main__":
    main()
