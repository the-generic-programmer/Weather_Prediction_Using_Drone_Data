import pandas as pd
import os
import joblib
import logging

def merge_predictions_with_logs(pred_csv, mavsdk_log_csv, output_csv):
    """
    Merge LSTM predictions with MAVSDK logger data on timestamp.
    Assumes both files have a 'timestamp' column.
    """
    pred_df = pd.read_csv(pred_csv)
    log_df = pd.read_csv(mavsdk_log_csv)
    merged = pd.merge_asof(pred_df.sort_values('timestamp'), log_df.sort_values('timestamp'), on='timestamp', direction='nearest')
    merged.to_csv(output_csv, index=False)
    logging.info(f"Merged predictions and logs saved to {output_csv}")

if __name__ == "__main__":
    # Example usage
    pred_csv = "predicted_wind.csv"  # Output from predict_lstm.py
    mavsdk_log_csv = "../MAVSDK_logger.csv"  # Adjust path as needed
    output_csv = "merged_predictions_logs.csv"
    merge_predictions_with_logs(pred_csv, mavsdk_log_csv, output_csv)
