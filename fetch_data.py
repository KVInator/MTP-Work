import yfinance as yf # type: ignore
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
    'BOSCHLTD.NS'
]

    fetch_and_save_data(tickers)
