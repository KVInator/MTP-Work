import yfinance as yf
import os
from datetime import datetime

def fetch_and_save_testing_data(tickers, folder='test_data/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    for ticker in tickers:
        print(f"Downloading {ticker} testing data...")
        data = yf.download(ticker, start='2023-01-01', end=end_date)
        data.to_csv(f'{folder}{ticker}_test.csv')
        print(f"{ticker} testing data saved to {folder}{ticker}_test.csv")

if __name__ == "__main__":
    tickers = tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 
    'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'WIPRO.NS', 
    'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'NTPC.NS', 'M&M.NS', 
    'NESTLEIND.NS', 'SUNPHARMA.NS', 'DRREDDY.NS', 
    'DIVISLAB.NS', 'GRASIM.NS', 'TATASTEEL.NS', 'BPCL.NS', 
    'ONGC.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS', 
    'ICICIBANK.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS', 'GAIL.NS', 'IOC.NS', 
    'VEDL.NS', 'SIEMENS.NS', 'AMBUJACEM.NS', 
    'BOSCHLTD.NS']
    
    fetch_and_save_testing_data(tickers)
