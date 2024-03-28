from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():    
    try:
        logging.info("Initializing Binance client...")
        client = Client()

        symbol = 'BTCUSDT'
        interval = client.KLINE_INTERVAL_15MINUTE
        file_path = 'data/btcusdt_klines_with_indicators.parquet'
        start_date = datetime.now() - timedelta(days=90)

        end_date = datetime.now()
        start_str = int(start_date.timestamp() * 1000)
        end_str = int(end_date.timestamp() * 1000)

        logging.info("Fetching historical klines...")
        klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str, limit=1000)
        
        logging.info("Converting klines to DataFrame...")
        klines_df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        klines_df[['Open', 'High', 'Low', 'Close', 'Volume']] = klines_df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)
        
        klines_df.ffill(inplace=True)
        klines_df = klines_df.drop(['Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], axis=1)
        klines_df.to_parquet(file_path)

        logging.info("Operation completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
