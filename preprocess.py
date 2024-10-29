import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_stock_data(files, seq_length=50):
    all_stock_sequences = []
    all_stock_returns = []
    dates = []

    scaler_open = StandardScaler()
    scaler_high = StandardScaler()
    scaler_low = StandardScaler()
    scaler_close = StandardScaler()
    scaler_volume = StandardScaler()

    all_open_data, all_high_data, all_low_data, all_close_data, all_volume_data = [], [], [], [], []
    for file_path in files:
        data = pd.read_csv(file_path)
        all_open_data.append(data[['Open']].values)
        all_high_data.append(data[['High']].values)
        all_low_data.append(data[['Low']].values)
        all_close_data.append(data[['Close']].values)
        all_volume_data.append(data[['Volume']].values)

    scaler_open.fit(np.vstack(all_open_data))
    scaler_high.fit(np.vstack(all_high_data))
    scaler_low.fit(np.vstack(all_low_data))
    scaler_close.fit(np.vstack(all_close_data))
    scaler_volume.fit(np.vstack(all_volume_data))

    stock_features_list = []
    stock_returns_list = []
    for file_path in files:
        data = pd.read_csv(file_path)
        dates.append(data['Date'].values)
        
        scaled_open = scaler_open.transform(data[['Open']].values)
        scaled_high = scaler_high.transform(data[['High']].values)
        scaled_low = scaler_low.transform(data[['Low']].values)
        scaled_close = scaler_close.transform(data[['Close']].values)
        scaled_volume = scaler_volume.transform(data[['Volume']].values)

        stock_features = np.hstack([scaled_open, scaled_high, scaled_low, scaled_close, scaled_volume])

        data['Return'] = data['Close'].pct_change().fillna(0)
        returns = data['Return'].values

        stock_sequences = []
        stock_targets = []
        for i in range(len(stock_features) - seq_length):
            stock_sequences.append(stock_features[i:i + seq_length])
            stock_targets.append(returns[i + seq_length])

        stock_features_list.append(np.array(stock_sequences))
        stock_returns_list.append(np.array(stock_targets))

    X_combined = np.stack(stock_features_list, axis=-1)  # Shape: [num_sequences, seq_length, num_assets * num_features]
    y_combined = np.column_stack(stock_returns_list)  # Shape: [num_sequences, num_assets]

    return X_combined, y_combined, dates, scaler_close, scaler_volume
