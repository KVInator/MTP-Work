import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_stock_data(files, seq_length=60, target_column='Close'):
    all_stock_features = []
    all_stock_returns = []
    dates = []

    for file_path in files:
        # Load the CSV file for each stock
        data = pd.read_csv(file_path)
        
        # Store the dates for alignment
        dates.append(data['Date'].values)
        
        # Normalize the 'Close' and 'Volume' using StandardScaler for this stock
        scaler_close = StandardScaler()
        scaler_volume = StandardScaler()
        
        # Normalize the 'Close' and 'Volume' features
        scaled_close = scaler_close.fit_transform(data[['Close']])
        scaled_volume = scaler_volume.fit_transform(data[['Volume']])

        # Combine scaled 'Close' and 'Volume' as features
        stock_features = np.hstack([scaled_close, scaled_volume])

        # Calculate log returns (log of percentage change in 'Close' prices)
        data['Log_Return'] = np.log(data[target_column] / data[target_column].shift(1))
        
        # Drop NaN rows created by the log return calculation
        data = data.dropna().reset_index(drop=True)

        # Collect log returns as the target (keep each stock's returns separately)
        scaled_returns = data['Log_Return'].values.reshape(-1, 1)  # Reshape to (len(data), 1)

        # Build sequence data for LSTM input (time series data)
        sequences = []
        returns = []
        for i in range(len(stock_features) - seq_length):
            sequences.append(stock_features[i:i + seq_length])  # Feature sequence
            returns.append(scaled_returns[i + seq_length - 1])  # Next period's return

        all_stock_features.append(np.array(sequences))  # Feature sequence
        all_stock_returns.append(np.array(returns))  # Log returns for this stock as the target

    # Stack sequences for all stocks vertically (as separate samples)
    X_combined = np.vstack(all_stock_features)

    # Stack returns for all stocks horizontally (each column is one stock's returns)
    y_combined = np.hstack(all_stock_returns)

    # Ensure feature matrix matches the number of return values
    min_length = min(X_combined.shape[0], y_combined.shape[0])

    # Trim both features and targets to the same length
    X_scaled = np.array(X_combined[:min_length, :])  # Adjust the feature matrix to match the target length
    y_scaled = np.array(y_combined[:min_length, :])  # Adjust the target matrix to match the feature length

    return X_scaled, y_scaled, dates[:min_length], scaler_close
