import yfinance as yf
import os

def fetch_and_save_data(tickers, folder='data/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for ticker in tickers:
        print(f"Downloading {ticker} data...")
        data = yf.download(ticker, start='2005-01-01', end='2023-01-01')
        data.to_csv(f'{folder}{ticker}.csv')
        print(f"{ticker} data saved to {folder}{ticker}.csv")

if __name__ == "__main__":
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 
               'HINDUNILVR.NS', 'ITC.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 
               'LT.NS', 'WIPRO.NS']
    fetch_and_save_data(tickers)
