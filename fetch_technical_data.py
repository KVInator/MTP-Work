import os
import pandas as pd
import numpy as np
import talib  # type: ignore  # For technical indicators

# Paths
DATA_FOLDER = "/Users/pranayvij/MTP-Work/data/"  
TECHNICAL_DATA_FOLDER = "/Users/pranayvij/MTP-Work/technical_data/"  

# Ensure technical data folder exists
if not os.path.exists(TECHNICAL_DATA_FOLDER):
    os.makedirs(TECHNICAL_DATA_FOLDER)

# List of tickers
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 
    'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'WIPRO.NS', 
    'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'NTPC.NS', 'M&M.NS', 
    'NESTLEIND.NS', 'SUNPHARMA.NS', 'DRREDDY.NS', 
    'DIVISLAB.NS', 'GRASIM.NS', 'TATASTEEL.NS', 'BPCL.NS', 
    'ONGC.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS', 
    'ICICIBANK.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS', 'GAIL.NS', 'IOC.NS', 
    'VEDL.NS', 'SIEMENS.NS', 'AMBUJACEM.NS', 'BOSCHLTD.NS'
]

# Function to compute technical indicators
def process_stock_data(ticker):
    file_path = os.path.join(DATA_FOLDER, f"{ticker}.csv")

    if not os.path.exists(file_path):
        print(f"Skipping {ticker}: Data file not found.")
        return None

    print(f"Processing {ticker}...")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Add 'Ticker' column
    df["Ticker"] = ticker

    # Compute Technical Indicators
    df["Daily Return"] = df["Adj Close"].pct_change()
    df["SMA_50"] = talib.SMA(df["Adj Close"], timeperiod=50)
    df["SMA_200"] = talib.SMA(df["Adj Close"], timeperiod=200)
    df["RSI"] = talib.RSI(df["Adj Close"], timeperiod=14)
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Adj Close"])
    df["Upper_Band"], df["Middle_Band"], df["Lower_Band"] = talib.BBANDS(df["Adj Close"])
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Adj Close"], timeperiod=14)
    df["ADX"] = talib.ADX(df["High"], df["Low"], df["Adj Close"], timeperiod=14)
    df["Stochastic_%K"], df["Stochastic_%D"] = talib.STOCH(df["High"], df["Low"], df["Adj Close"])
    df["Williams_%R"] = talib.WILLR(df["High"], df["Low"], df["Adj Close"], timeperiod=14)
    df["OBV"] = talib.OBV(df["Adj Close"], df["Volume"])
    df["Support_Level"] = df["Low"].rolling(window=20).min()
    df["Resistance_Level"] = df["High"].rolling(window=20).max()

    # Fill NaN values with 0 to avoid merge issues
    df.fillna(0, inplace=True)

    # Save processed data
    save_path = os.path.join(TECHNICAL_DATA_FOLDER, f"{ticker}_technical.csv")
    df.to_csv(save_path)

    print(f"✅ Saved processed data for {ticker} to {save_path}")

# Process all stocks
for ticker in tickers:
    process_stock_data(ticker)

print("✅ Technical data processing complete.")
