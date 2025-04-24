import os
import pandas as pd

# Paths
TECHNICAL_DATA_FOLDER = "/Users/pranayvij/MTP-Work/technical_data/"
CORRELATION_DATA_FOLDER = "/Users/pranayvij/MTP-Work/correlation_data/"

# Ensure correlation data folder exists
if not os.path.exists(CORRELATION_DATA_FOLDER):
    os.makedirs(CORRELATION_DATA_FOLDER)

# Load all stock technical data
all_data = {}
files = [f for f in os.listdir(TECHNICAL_DATA_FOLDER) if f.endswith(".csv")]

for file in files:
    file_path = os.path.join(TECHNICAL_DATA_FOLDER, file)
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    all_data[file.replace("_technical.csv", "")] = df["Adj Close"].pct_change()  

# Create correlation matrix
returns_df = pd.DataFrame(all_data)
correlation_matrix = returns_df.corr()

# Save results
correlation_output_path = os.path.join(CORRELATION_DATA_FOLDER, "stock_correlation.csv")
correlation_matrix.to_csv(correlation_output_path)

print(f"Stock correlation matrix saved to: {correlation_output_path}")
