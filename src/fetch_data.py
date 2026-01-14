import yfinance as yf
import pandas as pd
import os

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)

for ticker in stocks:
    df = yf.download(ticker, period="3y", interval="1d")
    df.reset_index(inplace=True)  # Move Date from index to column
    
    # Flexible column selection
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    columns_existing = [col for col in columns if col in df.columns]
    df = df[columns_existing]
    
    # Save clean CSV
    df.to_csv(f"{data_dir}/{ticker}_historical.csv", index=False)
    print(f"{ticker} historical data saved! Columns: {columns_existing}")
