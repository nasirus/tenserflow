import pandas as pd
import tensorflow as tf
from tools.prepare_data import prepare_data_with_indicators
import logging

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
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=((input_window_size, X.shape[2]))),
        
        # LSTM layers
        tf.keras.layers.LSTM(50, return_sequences=True, name='lstm_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.LSTM(100, return_sequences=False, name='lstm_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),
        
        # Dense layers for prediction
        tf.keras.layers.Dense(40, name='dense_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_3'),

        tf.keras.layers.Dense(future_window_size * 2, name='output_layer')  # For 10 days prediction, each with high and low
    ])

    print(model.summary())

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    logging.info("Compiling model...")
    
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())

    
    logging.info("Training model...")
    # Modify model.fit to include callbacks
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
    
    logging.info("Saving model...")
    # Save the model for future use
    model.save("stock_prediction_model.keras")
