import pandas as pd
from indicators import calculate_indicators

df = pd.read_parquet('data/btcusdt_klines_with_indicators.parquet')

# Preprocess the data
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')

df = calculate_indicators(df)

# Adjust display settings to show all rows and columns
pd.set_option('display.max_rows', None)  # None means show all rows
pd.set_option('display.max_columns', None)  # None means show all columns

print(df.tail())
print(df.columns.tolist())
print(len(df.columns.tolist()))
