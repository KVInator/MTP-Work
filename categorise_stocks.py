import os
import pandas as pd

# Paths
RESULTS_FOLDER = "/Users/pranayvij/MTP-Work/results/"
CATEGORIZED_RESULTS_FOLDER = "/Users/pranayvij/MTP-Work/categorized_results/"

# Ensure categorized results folder exists
if not os.path.exists(CATEGORIZED_RESULTS_FOLDER):
    os.makedirs(CATEGORIZED_RESULTS_FOLDER)

# Load final ranked stocks
file_path = os.path.join(RESULTS_FOLDER, "final_stock_selection.csv")
if not os.path.exists(file_path):
    print("❌ ERROR: final_stock_selection.csv not found. Run ranking script first.")
    exit()

df = pd.read_csv(file_path)

# Define selection criteria
def classify_risk(row):
    """Classifies stocks into Risk-Averse, Risk-Neutral, and Risk-Seeking categories."""
    if row["Volatility"] < df["Volatility"].quantile(0.3) and row["Sharpe Ratio"] > df["Sharpe Ratio"].quantile(0.7):
        return "Risk-Averse"
    elif df["Volatility"].quantile(0.3) <= row["Volatility"] <= df["Volatility"].quantile(0.7):
        return "Risk-Neutral"
    else:
        return "Risk-Seeking"

# Apply classification
df["Risk Profile"] = df.apply(classify_risk, axis=1)

# Select top 15 stocks per category
risk_averse_df = df[df["Risk Profile"] == "Risk-Averse"].nlargest(15, "Final Score")
risk_neutral_df = df[df["Risk Profile"] == "Risk-Neutral"].nlargest(15, "Final Score")
risk_seeking_df = df[df["Risk Profile"] == "Risk-Seeking"].nlargest(15, "Final Score")

# Save results
risk_averse_df.to_csv(os.path.join(CATEGORIZED_RESULTS_FOLDER, "risk_averse_stocks.csv"), index=False)
risk_neutral_df.to_csv(os.path.join(CATEGORIZED_RESULTS_FOLDER, "risk_neutral_stocks.csv"), index=False)
risk_seeking_df.to_csv(os.path.join(CATEGORIZED_RESULTS_FOLDER, "risk_seeking_stocks.csv"), index=False)

print("✅ Stocks categorized and saved:")
print(f"- Risk-Averse: {len(risk_averse_df)} stocks saved.")
print(f"- Risk-Neutral: {len(risk_neutral_df)} stocks saved.")
print(f"- Risk-Seeking: {len(risk_seeking_df)} stocks saved.")
