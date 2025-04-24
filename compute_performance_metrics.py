import os
import pandas as pd
import numpy as np

# Paths
TECHNICAL_DATA_FOLDER = "/Users/pranayvij/MTP-Work/technical_data/"
PERFORMANCE_DATA_FOLDER = "/Users/pranayvij/MTP-Work/performance_data/"

# Ensure folder exists
if not os.path.exists(PERFORMANCE_DATA_FOLDER):
    os.makedirs(PERFORMANCE_DATA_FOLDER)

# Risk-Free Rate (Assumed 5% annually, converted to daily)
risk_free_rate = 0.05 / 252  

metrics_data = []

# Load stock data
files = [f for f in os.listdir(TECHNICAL_DATA_FOLDER) if f.endswith(".csv")]

for file in files:
    file_path = os.path.join(TECHNICAL_DATA_FOLDER, file)
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    df["Daily Return"] = df["Adj Close"].pct_change()

    # Compute Sharpe Ratio
    mean_return = df["Daily Return"].mean()
    std_dev = df["Daily Return"].std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan

    # Compute Sortino Ratio (Using only downside deviation)
    downside_returns = df["Daily Return"][df["Daily Return"] < 0]
    downside_std_dev = downside_returns.std()
    sortino_ratio = (mean_return - risk_free_rate) / downside_std_dev if downside_std_dev != 0 else np.nan

    # Store metrics
    metrics_data.append({
        "Ticker": file.replace("_technical.csv", ""),
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
    })

# Convert to DataFrame & Save
metrics_df = pd.DataFrame(metrics_data)
metrics_output_path = os.path.join(PERFORMANCE_DATA_FOLDER, "performance_metrics.csv")
metrics_df.to_csv(metrics_output_path, index=False)

print(f"Performance metrics saved to: {metrics_output_path}")
