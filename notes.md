/home/yadu/Desktop/Weather_Prediction_Using_Drone_Data-main/.venv/bin/python /home/yadu/Desktop/Weather_Prediction_Using_Drone_Data-main/ml_weathe
r/train_model.py
Loading historical weather data...
Preparing features...
Training model...

Evaluating model performance...

Training Set Metrics:
RMSE: 0.91C
R2 Score: 0.979

Test Set Metrics:
RMSE: 1.89C
R2 Score: 0.910

Saving model...
Model saved in 'models' directory

RSME IS RELATIVELY LOW FOR THIS TRAINED MODEL...

///

currently the thing fetches data for london

///

sudo docker run -d -p 3000:3000 \
  -v /home/aaditya/Desktop/Weather_Prediction_Using_Drone_Data-main/database:/data \
  --name metabase metabase/metabase

  ///

  