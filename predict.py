import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tools.prepare_data import prepare_data_with_indicators
from load_data import load_data
from model_train import build_model
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_window_size = 168
future_window_size = 72

# Initialize the Binance client
client = Client()
# Assuming you have an API client for fetching historical data; you might need to import or configure it

def fetch_historical_prices(symbol, interval, days_ago):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_ago)
    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b, %Y %H:%M:%S"), end_time.strftime("%d %b, %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    # Convert necessary columns to numeric types for calculations
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)
    # Exclude the last row from the data
    data = data.iloc[:-1]
    return data


# Prepare new data for prediction
def prepare_new_data_for_prediction(new_df):
    # Assume calculate_indicators and clean_and_prepare_data are implemented as per the earlier script
    X, y, scaler_features, scaler_targets = prepare_data_with_indicators(new_df, input_window_size, future_window_size)
    return X, y, scaler_features, scaler_targets

# Function to make predictions
def make_predictions(model, new_data_frame):
    X, _, _, scaler_targets = prepare_new_data_for_prediction(new_data_frame)
    
    predictions_scaled = model.predict(X[-input_window_size:])
        
    n_samples = predictions_scaled.shape[0]
    n_features = 2

    # Reshape predictions to (n_samples * future_time_steps, n_features)
    predictions_reshaped = predictions_scaled.reshape(-1, n_features)

    # Apply inverse transform
    predictions_inversed = scaler_targets.inverse_transform(predictions_reshaped)

    # Optionally, reshape back to (n_samples, future_time_steps * n_features)
    predictions_final = predictions_inversed.reshape(n_samples, future_window_size * n_features)

    # Assuming predictions_final is in shape (n_samples, future_time_steps * n_features)
    # And you have 10 future time steps and 2 features (low and high)
    last_prediction = predictions_final[-1]  # Get the last set of predictions
    return last_prediction  # or predictions_inversed if you did inverse scaling

logging.info("Start loading data from exchange...")
load_data()

logging.info("Start building and training model...")
build_model(input_window_size=input_window_size,future_window_size=future_window_size)

logging.info("Loading model from file...")
model = tf.keras.models.load_model('models/stock_prediction_model.keras')

logging.info("Fetching historical data...")
# Fetch new data (you'll need to implement or connect fetch_historical_prices to a real data source)
new_data_frame = fetch_historical_prices('BTCUSDT', client.KLINE_INTERVAL_1HOUR, 90)

logging.info("Making prediction...")
# Make predictions
predictions = make_predictions(model=model, new_data_frame=new_data_frame)

# Extract lows and highs for the last time step
lows_last = predictions[::2]  # Taking every other value starting from 0 for lows
highs_last = predictions[1::2]  # Taking every other value starting from 1 for highs

# Displaying the lows and highs for the last prediction
logging.info(f"Last Prediction Lows: {lows_last}")
logging.info(f"Last Prediction Highs: {highs_last}")

# Assuming `df` is your DataFrame containing the historical data
# Selecting the last 100 rows and specifically including 'Close' and 'Open time' columns
last_100 = new_data_frame[['Close', 'Open time']].tail(input_window_size)

# Converting 'Open time' to datetime format
last_100['Open time'] = pd.to_datetime(last_100['Open time'], unit='ms')

# Setting 'Open time' as the index
last_100 = last_100.set_index('Open time')

# Assuming df has an 'Open time' column in datetime format
last_date = pd.to_datetime(new_data_frame['Open time'].iloc[-1], unit='ms')

# Generate future timestamps
# Assuming last_date is the last date in your historical data and is already a datetime object
future_dates = pd.date_range(start=last_date, freq=pd.Timedelta(hours=1), periods=future_window_size)

plt.figure(figsize=(14, 7))

# Plotting the historical close prices
plt.plot(last_100.index, last_100['Close'], label='Historical Close Price', color='skyblue')

# Plotting the predicted lows and highs with correct timestamps
plt.plot(future_dates, lows_last, label='Predicted Lows', marker='o', linestyle='dashed', color='green')
plt.plot(future_dates, highs_last, label='Predicted Highs', marker='o', linestyle='dashed', color='red')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Last 100 Historical Close Prices and Predicted Lows & Highs')
plt.legend()

# Formatting the date on the x-axis for better readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate() # Rotation

plt.show()