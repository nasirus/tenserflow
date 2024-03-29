import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tools.prepare_data import prepare_data_with_indicators
import logging
import os

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model definition
def build_model(input_window_size = 96, future_window_size = 24, epochs=100, batch_size=20):
    # Assuming df is your DataFrame containing historical price data and indicators

    logging.info("Loading data from file...")
        
    df = pd.read_parquet('data/btcusdt_klines_with_indicators.parquet')
    df = df[:len(df)-1]

    logging.info("Preparing data with indicators...")

    X, y, _, _ = prepare_data_with_indicators(df, input_window_size, future_window_size)

    logging.info("Building model...")
    
    model = Sequential([
        Input(shape=((input_window_size, X.shape[2]))),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(512, return_sequences=False),
        Dropout(0.2),
        Dense(400, activation='relu'),
        Dropout(0.2),
        Dense(future_window_size * 2, name='output_layer')  # For 10 days prediction, each with high and low
    ])

    print(model.summary())

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    logging.info("Compiling model...")
    
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())

    
    logging.info("Training model...")
    # Modify model.fit to include callbacks
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2
              , callbacks=[early_stopping]
              )
    
    logging.info("Saving model...")
    model_directory = 'models'  # Define the directory to store scaler files
    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Save the model for future use
    model.save(os.path.join(model_directory, 'stock_prediction_model.keras'))
