import pandas as pd
import ta

def calculate_indicators(df):

    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    return df
"""
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    df['SMA14'] = ta.trend.SMAIndicator(df['Close'], window=14).sma_indicator()
    df['SMA24'] = ta.trend.SMAIndicator(df['Close'], window=24).sma_indicator()
    df['SMA48'] = ta.trend.SMAIndicator(df['Close'], window=48).sma_indicator()
    
    df['EMA14'] = ta.trend.EMAIndicator(df['Close'], window=14).ema_indicator()
    df['EMA24'] = ta.trend.EMAIndicator(df['Close'], window=24).ema_indicator()
    df['EMA48'] = ta.trend.EMAIndicator(df['Close'], window=48).ema_indicator()

    # RSI range categorization
    df['RSI_range'] = df['RSI'].apply(lambda x: 0 if x < 30 else (1 if x <= 70 else 2))
    
    df['MA_range_price_14'] = 0  # Default value
    df.loc[df['Close'] >= df['SMA14'], 'MA_range_price_14'] = 1
    df.loc[df['Close'] < df['SMA14'], 'MA_range_price_14'] = 0
    
    df['MA_range_price_24'] = 0  # Default value
    df.loc[df['Close'] >= df['SMA24'], 'MA_range_price_24'] = 1
    df.loc[df['Close'] <  df['SMA24'], 'MA_range_price_24'] = 0
    
    df['MA_range_price_48'] = 0  # Default value
    df.loc[df['Close'] >= df['SMA48'], 'MA_range_price_48'] = 1
    df.loc[df['Close'] < df['SMA48'], 'MA_range_price_48'] = 0

    df['Movement'] = 1  # Default value
    df.loc[df['Close'] >= df['Close'].shift(1), 'Movement'] = 1
    df.loc[df['Close'] < df['Close'].shift(1), 'Movement'] = 0

    df['SAR_psar_down_indicator'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar_down_indicator()
    df['SAR_psar_up_indicator'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar_up_indicator()
    
    df['MACD_diff'] = ta.trend.macd_diff(df['Close'])
    
    df['BOL_h'] = ta.volatility.bollinger_hband_indicator(df['Close'])
    df['BOL_l'] = ta.volatility.bollinger_lband_indicator(df['Close'])
    df['BOL_p'] = ta.volatility.bollinger_pband(df['Close'])

    df['PPO_s'] = ta.momentum.PercentagePriceOscillator(close=df['Close']).ppo_signal()

    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['ADX_neg'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_neg()
    df['ADX_pos'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_pos()

    df['Stoch_Osc_signal'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'],volume=df['Volume']).on_balance_volume()

    df['PVO_s'] = ta.momentum.PercentageVolumeOscillator(volume=df['Volume']).pvo_signal()
    
    df = df.drop(['RSI', 'SMA14', 'SMA24', 'SMA48', 'EMA14', 'EMA24', 'EMA48', 'RSI_range'], axis=1)
"""
