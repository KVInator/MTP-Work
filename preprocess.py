import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_stock_data(files, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'], target_column='Adj Close', window_size=60):
    all_data = []
    
    for file_path in files:
        # Load the CSV file
        data = pd.read_csv(file_path)
        data = data[feature_columns + [target_column]].dropna()
        all_data.append(data)

    # Concatenate data from all stocks
    combined_data = pd.concat(all_data, axis=0).reset_index(drop=True)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)

    X_scaled = []
    y_scaled = []

    # Create windows of time-series data for training the model
    for i in range(window_size, len(scaled_data)):
        X_scaled.append(scaled_data[i-window_size:i, :-1])  # Use all features except target
        y_scaled.append(scaled_data[i, -1])  # Use the adjusted close as target

    X_scaled = np.array(X_scaled)
    y_scaled = np.array(y_scaled)

    return X_scaled, y_scaled, scaler
