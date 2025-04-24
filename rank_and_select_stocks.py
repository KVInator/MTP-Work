import os
import pandas as pd
import numpy as np

# Paths
FUNDAMENTAL_DATA_FOLDER = "/Users/pranayvij/MTP-Work/fundamental_data/"
TECHNICAL_DATA_FOLDER = "/Users/pranayvij/MTP-Work/technical_data/"
PERFORMANCE_DATA_FOLDER = "/Users/pranayvij/MTP-Work/performance_data/"
RESULTS_FOLDER = "/Users/pranayvij/MTP-Work/results/"

# Ensure results folder exists
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

print("🔄 Loading Fundamental Data...")

# Load Fundamental Data
fundamental_files = [f for f in os.listdir(FUNDAMENTAL_DATA_FOLDER) if f.endswith("_fundamental.csv")]
if not fundamental_files:
    print("❌ ERROR: No fundamental data files found.")
    exit()

fundamental_df = pd.concat([pd.read_csv(os.path.join(FUNDAMENTAL_DATA_FOLDER, f)) for f in fundamental_files])

print(f"✅ Loaded {len(fundamental_df)} fundamental records.")

print("🔄 Loading Technical Data...")

# Load Technical Data (Aggregated)
technical_files = [f for f in os.listdir(TECHNICAL_DATA_FOLDER) if f.endswith("_technical.csv")]
if not technical_files:
    print("❌ ERROR: No technical data files found.")
    exit()

# Aggregate technical indicators for each stock
technical_data = []
for file in technical_files:
    file_path = os.path.join(TECHNICAL_DATA_FOLDER, file)
    df = pd.read_csv(file_path, parse_dates=["Date"])

    # Compute average values across all dates
    stock_data = {
        "Ticker": df["Ticker"].iloc[0],
        "Avg RSI": df["RSI"].mean(),
        "Avg ATR": df["ATR"].mean(),
        "Avg ADX": df["ADX"].mean(),
        "Avg MACD": df["MACD"].mean(),
        "Avg Stochastic_K": df["Stochastic_%K"].mean(),
        "Avg Stochastic_D": df["Stochastic_%D"].mean(),
        "Volatility": df["Daily Return"].std(),  # Measures risk
    }
    technical_data.append(stock_data)

technical_df = pd.DataFrame(technical_data)
print(f"✅ Loaded and aggregated technical data for {len(technical_df)} stocks.")

print("🔄 Loading Performance Data...")

# Load Performance Data
performance_file = os.path.join(PERFORMANCE_DATA_FOLDER, "performance_metrics.csv")
if not os.path.exists(performance_file):
    print("❌ ERROR: Performance metrics file not found.")
    exit()

performance_df = pd.read_csv(performance_file)

print(f"✅ Loaded {len(performance_df)} performance records.")

# Merge All Data
merged_df = technical_df.merge(fundamental_df, on="Ticker", how="inner").merge(performance_df, on="Ticker", how="inner")

if merged_df.empty:
    print("❌ ERROR: Merged dataset is empty. Check input files.")
    exit()

print(f"✅ Merged dataset contains {len(merged_df)} records.")

print("🔄 Computing Final Score...")

# Compute Final Score (Long-Term Metrics)
merged_df["Final Score"] = (
    merged_df["Sharpe Ratio"].mean() * 0.3 +  
    merged_df["Sortino Ratio"].mean() * 0.2 +  
    (1 / merged_df["P/E Ratio"].median()) * 0.1 +  
    merged_df["Avg RSI"] * 0.1 +  
    merged_df["Avg ADX"] * 0.1 +  
    (1 / merged_df["Debt-to-Equity"].median()) * 0.1 -  
    merged_df["Volatility"] * 0.1  # Penalizing high volatility
)

print("✅ Final Score Computed.")

# Sort Stocks by Score
ranked_df = merged_df.sort_values(by="Final Score", ascending=False)

# Save Final Ranked Stocks
ranking_output_path = os.path.join(RESULTS_FOLDER, "final_stock_selection.csv")
ranked_df.to_csv(ranking_output_path, index=False)

print(f"✅ Final stock selection saved to: {ranking_output_path}")
print(ranked_df.head(10))  # Show top 10 stocks
