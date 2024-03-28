import numpy as np
from sklearn.preprocessing import RobustScaler
from tools.indicators import calculate_indicators
from tools.clean_df import clean_and_prepare_data


# Data preparation function including indicators
def prepare_data_with_indicators(df, input_window_size=60, future_window_size=10, scaler=RobustScaler()):
    df = calculate_indicators(df)
    df = clean_and_prepare_data(df)
    feature_columns = [col for col in df.columns if col not in ['Open time', 'Open','Low', 'High', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']]
    target_columns = ['Low', 'High']
    
    data_features = df[feature_columns].values
    data_targets = df[target_columns].values

    scaler_features = scaler
    scaler_targets = scaler

    scaled_features = scaler_features.fit_transform(data_features)
    scaled_targets = scaler_targets.fit_transform(data_targets)

    X, y = [], []
    for i in range(input_window_size, len(df) - future_window_size):
        X.append(scaled_features[i - input_window_size:i])
        y.append(scaled_targets[i:i + future_window_size].flatten())
    return np.array(X), np.array(y), scaler_features, scaler_targets


def prepare_predict_with_indicators(df, input_window_size=60, scaler=RobustScaler()):
    # Calculate indicators and clean the data
    df = calculate_indicators(df)
    df = clean_and_prepare_data(df)

    # Define feature columns, excluding non-feature columns
    feature_columns = [col for col in df.columns if col not in ['Open time', 'Open', 'Low', 'High', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']]
    
    # Extract features
    data_features = df[feature_columns].values

    # Initialize and fit scalers
    scaler_features = scaler
    scaled_features = scaler_features.fit_transform(data_features)

    # Prepare the last 'input_window_size' features as input for prediction
    X = scaled_features[-input_window_size:].reshape(1, input_window_size, -1) # Reshape for model input

    return X, scaler_features