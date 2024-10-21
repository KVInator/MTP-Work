import pandas as pd
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
        
        # Calculate arithmetic returns (percentage change)
        data['Return'] = data[target_column].pct_change()  # (P_t / P_{t-1}) - 1
        
        # Drop NaN rows created by the return calculation
        data = data.dropna().reset_index(drop=True)

        # Collect returns as the target (keep each stock's returns separately)
        returns = data['Return'].values.reshape(-1, 1)  # Reshape to (len(data), 1)

        # Build sequence data for LSTM input (time series data)
        sequences = []
        returns_seq = []
        for i in range(len(data) - seq_length):
            sequences.append(data[target_column].values[i:i + seq_length].reshape(-1, 1))  # Use 'Close' prices as features
            returns_seq.append(returns[i + seq_length - 1])  # Next period's return

        all_stock_features.append(np.array(sequences))  # Feature sequence
        all_stock_returns.append(np.array(returns_seq))  # Returns for this stock as the target

    # Stack sequences for all stocks vertically (as separate samples)
    X_combined = np.vstack(all_stock_features)

    # Stack returns for all stocks horizontally (each column is one stock's returns)
    y_combined = np.hstack(all_stock_returns)

    # Ensure feature matrix matches the number of return values
    min_length = min(X_combined.shape[0], y_combined.shape[0])

    # Trim both features and targets to the same length
    X_scaled = np.array(X_combined[:min_length, :])  # Adjust the feature matrix to match the target length
    y_scaled = np.array(y_combined[:min_length, :])  # Adjust the target matrix to match the feature length

    return X_scaled, y_scaled, dates[:min_length]
