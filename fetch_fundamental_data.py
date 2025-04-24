import os
import pandas as pd
import yfinance as yf # type: ignore

# Paths
FUNDAMENTAL_DATA_FOLDER = "/Users/pranayvij/MTP-Work/fundamental_data/"  

# Ensure folder exists
if not os.path.exists(FUNDAMENTAL_DATA_FOLDER):
    os.makedirs(FUNDAMENTAL_DATA_FOLDER)

# List of tickers (Same as in `fetch_technical_data.py`)
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

# Fetch Fundamental Data
for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info

    data = {
        "Ticker": ticker,
        "P/E Ratio": info.get("trailingPE"),
        "P/B Ratio": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Earnings Growth": info.get("earningsGrowth"),
        "Revenue Growth": info.get("revenueGrowth"),
        "Debt-to-Equity": info.get("debtToEquity"),
    }

    df = pd.DataFrame([data])
    save_path = os.path.join(FUNDAMENTAL_DATA_FOLDER, f"{ticker}_fundamental.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved fundamental data for {ticker} at {save_path}")

print("Fundamental data processing complete.")
